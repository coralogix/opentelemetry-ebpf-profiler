// SPDX-License-Identifier: Apache-2.0

package cupti // import "go.opentelemetry.io/ebpf-profiler/gpu/cupti"

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"time"

	cebpf "github.com/cilium/ebpf"

	"go.opentelemetry.io/ebpf-profiler/internal/log"
	"go.opentelemetry.io/ebpf-profiler/libpf"
	"go.opentelemetry.io/ebpf-profiler/periodiccaller"
	"go.opentelemetry.io/ebpf-profiler/reporter"
	"go.opentelemetry.io/ebpf-profiler/reporter/samples"
)

// RunConfig wires Run to the loaded eBPF collection and the reporter.
type RunConfig struct {
	Progs          map[string]*cebpf.Program // tracer.EBPFPrograms()
	Maps           map[string]*cebpf.Map     // tracer.EBPFMaps()
	Reporter       TraceReporter
	Registrar      reporter.ProbeRegistrar
	AllocateOrigin func() libpf.Origin
}

// Handle exposes the hooks the agent wires into its trace pipeline, and the
// shutdown entry point.
type Handle struct {
	src     *Source
	matcher *Matcher
	pidCh   chan int
}

// OnNewPID is for the tracer's PID-event hook. Non-blocking: the scan runs on
// the rescan goroutine so a slow attach never stalls the trace pipeline (a
// dropped notification is covered by the next periodic rescan).
func (h *Handle) OnNewPID(pid libpf.PID) {
	select {
	case h.pidCh <- int(pid):
	default:
	}
}

// Stop detaches all probes, waits for the drains and flushes the remaining
// deltas. Call before stopping the reporter so the final flush can still be
// exported.
func (h *Handle) Stop() {
	_ = h.src.Stop() // quiesce first: events folded after a flush would be lost
	h.matcher.Flush()
}

// OnLaunchTrace is for the process manager's TRACE_GPU observer.
func (h *Handle) OnLaunchTrace(trace *libpf.Trace, meta *samples.TraceEventMeta) {
	h.matcher.OnLaunchTrace(trace, meta)
}

// Run starts GPU profiling: registers the sample-type origins, starts the
// ringbuf drains, the per-PID attach/reconcile rescan and the flush loop.
// ctx cancellation stops the rescan/flush/busy loops; call Handle.Stop to
// detach the probes, stop the drains and flush the final deltas.
func Run(ctx context.Context, cfg RunConfig) (*Handle, error) {
	if cfg.Progs == nil || cfg.Maps == nil || cfg.Reporter == nil ||
		cfg.Registrar == nil || cfg.AllocateOrigin == nil {
		return nil, fmt.Errorf("gpu/cupti: all RunConfig fields are required")
	}
	var origins Origins
	for _, spec := range []struct {
		dst  *libpf.Origin
		typ  string
		unit string
	}{
		{&origins.KernelTime, "gpu_kernel_time", "nanoseconds"},
		{&origins.MemTime, "gpu_mem_time", "nanoseconds"},
		{&origins.MemBytes, "gpu_mem_bytes", "bytes"},
		{&origins.BusyTime, "gpu_busy_time", "nanoseconds"},
		{&origins.UVMBytes, "gpu_uvm_bytes", "bytes"},
		{&origins.UVMFaults, "gpu_uvm_faults", "count"},
		{&origins.AllocBytes, "gpu_alloc_bytes", "bytes"},
		{&origins.StallSamples, "gpu_stall_samples", "count"},
		{&origins.NCCLTime, "gpu_nccl_time", "nanoseconds"},
		{&origins.NCCLBytes, "gpu_nccl_bytes", "bytes"},
		{&origins.APITime, "gpu_api_time", "nanoseconds"},
	} {
		*spec.dst = cfg.AllocateOrigin()
		if err := cfg.Registrar.RegisterProbeOrigin(*spec.dst, samples.ProbeOriginMetadata{
			Typ:          spec.typ,
			Unit:         spec.unit,
			ReportValues: true,
		}); err != nil {
			return nil, fmt.Errorf("register %s origin: %w", spec.typ, err)
		}
	}

	matcher := NewMatcher(cfg.Reporter, origins, NewNameCache())
	src, err := New(Config{
		Progs:    cfg.Progs,
		Maps:     cfg.Maps,
		OnTiming: matcher.OnTiming,
		OnMem:    matcher.OnMem,
		OnEvent:  matcher.OnEvent,
		// Shim-side failures (e.g. another CUPTI subscriber) are rare and
		// high-signal.
		OnError: func(e ShimError) {
			log.Warnf("gpu/cupti: shim error in pid %d (code %d): %s", e.PID, e.Code, e.Msg)
		},
		OnPIDGone: func(pid int) { matcher.PruneDeadPIDs([]int{pid}) },
	})
	if err != nil {
		return nil, err
	}
	if err := src.Start(); err != nil {
		return nil, err
	}

	// The shim is dlopen'd at cuInit, after exec, so the PID-event hook alone
	// races the injection; the rescan catches processes once the shim is
	// mapped and reconciles exited PIDs.
	rescan := func() {
		pids := listPIDs()
		if len(pids) == 0 {
			return // /proc read failure must not be mistaken for "all dead"
		}
		live := make(map[int]struct{}, len(pids))
		for _, p := range pids {
			live[p] = struct{}{}
			src.OnNewPID(p)
		}
		if dead := src.Reconcile(live); len(dead) > 0 {
			// Export accumulated deltas before dropping the dead PIDs' state,
			// or a short-lived job's final interval would be lost.
			matcher.Flush()
			matcher.PruneDeadPIDs(dead)
		}
		// Caches can be repopulated for dead PIDs by late ringbuf events and
		// hold entries for never-attached PIDs (busy sampling, fallbacks);
		// sweep them against the live set.
		matcher.SweepCaches(live)
	}
	// PID-event notifications and the periodic rescan share one goroutine so
	// attaches are serialized off the trace pipeline.
	pidCh := make(chan int, 256)
	go func() {
		t := time.NewTicker(2 * time.Second)
		defer t.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case pid := <-pidCh:
				src.OnNewPID(pid)
			case <-t.C:
				rescan()
			}
		}
	}()
	periodiccaller.Start(ctx, 5*time.Second, matcher.Flush)

	// Per-process GPU utilization via NVML: covers CUDA processes without the
	// shim injected, at coarse granularity.
	if bp := NewBusyPoller(func(pid uint32, busyNs int64) {
		matcher.AddSample(pid, []string{"gpu:busy"}, origins.BusyTime, busyNs)
	}); bp != nil {
		periodiccaller.Start(ctx, bp.interval, bp.poll)
	} else {
		log.Info("gpu/cupti: nvidia-smi not available (not in PATH, and no " +
			"host-rootfs copy reachable via nsenter); GPU busy sampling disabled")
	}

	return &Handle{src: src, matcher: matcher, pidCh: pidCh}, nil
}

// listPIDs enumerates current process IDs from /proc.
func listPIDs() []int {
	ents, err := os.ReadDir("/proc")
	if err != nil {
		return nil
	}
	pids := make([]int, 0, len(ents))
	for _, e := range ents {
		// bitSize 31: PIDs always fit (kernel max is 2^22), and the result is
		// provably in range for both int and the downstream uint32 conversions.
		if pid, err := strconv.ParseUint(e.Name(), 10, 31); err == nil {
			pids = append(pids, int(pid))
		}
	}
	return pids
}
