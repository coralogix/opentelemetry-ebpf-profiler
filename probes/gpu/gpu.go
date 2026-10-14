// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

// Package gpu implements GPU profiling as a probe. Workloads emit the otel_gpu
// USDT probes defined in otel_gpu_usdt.h through an injected producer, such as
// a CUPTI library loaded with CUDA_INJECTION64_PATH. The probe attaches to
// those probes per process, captures the host stack at each GPU call site and
// attributes the GPU time measured later to that stack.
//
// An OTel config using this probe could look like this:
//
//	receivers:
//	  ebpf_profiler:
//	    probes:
//	      - gpu
//
//	extensions:
//	  gpu: {}
package gpu // import "go.opentelemetry.io/ebpf-profiler/probes/gpu"

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	cebpf "github.com/cilium/ebpf"
	"github.com/cilium/ebpf/link"
	"github.com/cilium/ebpf/ringbuf"

	"go.opentelemetry.io/ebpf-profiler/internal/log"
	"go.opentelemetry.io/ebpf-profiler/libpf"
	"go.opentelemetry.io/ebpf-profiler/reporter/samples"
	"go.opentelemetry.io/ebpf-profiler/tracer"
	"go.opentelemetry.io/ebpf-profiler/usdt"
	"go.opentelemetry.io/ebpf-profiler/util"
)

// Config is the configuration of the gpu probe.
type Config struct{}

// launchProfileType counts GPU call sites per host stack. PreHandleTrace leaves
// the sample weight of the launch as the trace value.
var launchProfileType = &samples.TypeMetadata{
	SampleType:   "gpu_launches",
	SampleUnit:   "count",
	ReportValues: true,
}

// usdtPrograms maps each otel_gpu probe name to the eBPF program consuming it.
var usdtPrograms = map[string]string{
	"ready":           "uprobe__gpu_ready",
	"on_launch":       "uprobe__gpu_launch",
	"kernel_executed": "uprobe__gpu_kernel_executed",
	"gpu_mem":         "uprobe__gpu_mem",
	"gpu_event":       "uprobe__gpu_event",
	"error":           "uprobe__gpu_error",
}

// eventsMap carries the GPU records from eBPF.
const eventsMap = "gpu_events"

type probe struct {
	config Config

	// discoverer is only used from Attach, which the process manager
	// serializes.
	discoverer *usdt.Discoverer

	// programs holds the loaded eBPF programs by name; set once by Load.
	programs map[string]*cebpf.Program

	launchOrigin uint16
	// launch is the launch whose trace is being symbolized. PreHandleTrace
	// sets it for PostHandleTrace, both called from the trace handling loop.
	launch struct {
		pid           libpf.PID
		correlationID uint32
	}

	matcher *matcher
	names   *nameCache
	events  *cebpf.Map
	reader  *ringbuf.Reader
	drained sync.WaitGroup

	// mu serializes attaching with Unload, which closes the programs.
	mu       sync.Mutex
	unloaded bool
	// links holds the uprobe links of each process, per attached file.
	links map[libpf.PID]map[util.OnDiskFileIdentifier][]link.Link
}

// New returns the gpu probe.
func New(config Config) (tracer.Probe, error) {
	discoverer, err := usdt.NewDiscoverer()
	if err != nil {
		return nil, err
	}
	return &probe{
		config:     config,
		discoverer: discoverer,
		names:      newNameCache(),
		links:      make(map[libpf.PID]map[util.OnDiskFileIdentifier][]link.Link),
	}, nil
}

func (p *probe) String() string {
	return "gpu"
}

// Load implements tracer.Probe.
func (p *probe) Load(_ context.Context, reg tracer.ProbeRegistrar, probeCtx *tracer.ProbeContext) error {
	launchOrigin, err := reg.Register(launchProfileType)
	if err != nil {
		return fmt.Errorf("registering GPU launch origin: %w", err)
	}
	p.launchOrigin = launchOrigin
	if p.matcher, err = newMatcher(probeCtx.ProcessMeta, p.names); err != nil {
		return err
	}

	programNames := make([]string, 0, len(usdtPrograms))
	loaders := make([]tracer.ProgLoaderHelper, 0, len(usdtPrograms))
	for _, name := range usdtPrograms {
		programNames = append(programNames, name)
		loaders = append(loaders, tracer.ProgLoaderHelper{
			Name: name, NoTailCallTarget: true, Enable: true,
		})
	}
	coll, err := probeCtx.CollectionSpecWith(
		[]string{eventsMap},
		programNames,
		[]string{"origin_id_gpu_launch"},
	)
	if err != nil {
		return err
	}
	if err := coll.Variables["origin_id_gpu_launch"].Set(launchOrigin); err != nil {
		return fmt.Errorf("set origin_id_gpu_launch: %w", err)
	}

	if p.events, err = cebpf.NewMap(coll.Maps[eventsMap]); err != nil {
		return fmt.Errorf("creating %s: %w", eventsMap, err)
	}
	if err := probeCtx.RewriteMaps(coll, map[string]*cebpf.Map{eventsMap: p.events}); err != nil {
		_ = p.events.Close()
		return err
	}
	programs := make(map[string]*cebpf.Program)
	if err := probeCtx.LoadProbeUnwinders(coll, programs, loaders, 0); err != nil {
		_ = p.events.Close()
		return err
	}
	p.programs = programs

	if p.reader, err = ringbuf.NewReader(p.events); err != nil {
		p.closePrograms()
		_ = p.events.Close()
		return fmt.Errorf("reading %s: %w", eventsMap, err)
	}
	p.drained.Add(1)
	go p.drain()

	if err := probeCtx.AddAttacher(p); err != nil {
		p.stopDrain()
		p.closePrograms()
		_ = p.events.Close()
		return fmt.Errorf("registering process attacher: %w", err)
	}
	return nil
}

// drain forwards the GPU records from eBPF until the reader is closed.
func (p *probe) drain() {
	defer p.drained.Done()
	var record ringbuf.Record
	errorLogged := false
	for {
		if err := p.reader.ReadInto(&record); err != nil {
			if errors.Is(err, ringbuf.ErrClosed) {
				return
			}
			if !errorLogged {
				log.Warnf("gpu: reading GPU records: %v", err)
				errorLogged = true
			}
			// Avoid spinning on a persistent error.
			time.Sleep(10 * time.Millisecond)
			continue
		}
		p.handleRecord(record.RawSample)
	}
}

func (p *probe) handleRecord(b []byte) {
	recordType, pid, ok := recordHeader(b)
	if !ok {
		return
	}
	now := time.Now()
	switch recordType {
	case recordKernel:
		if rec, ok := decodeKernelRecord(pid, b); ok {
			p.matcher.addWork(kernelWork(rec, p.names.resolve(pid, rec.namePtr)), now)
		}
	case recordMemcpy:
		if rec, ok := decodeMemcpyRecord(pid, b); ok {
			p.matcher.addWork(memcpyWork(rec), now)
		}
	case recordEvent:
		rec, ok := decodeEventRecord(pid, b)
		if !ok {
			return
		}
		if work, ok := eventWork(rec); ok {
			p.matcher.addWork(work, now)
		}
	case recordError:
		// Producers re-emit errors periodically, which bounds this logging.
		if rec, ok := decodeErrorRecord(pid, b); ok {
			log.Warnf("gpu: producer error in PID %d (code %d): %s", rec.pid, rec.code, rec.message)
		}
	}
}

func (p *probe) stopDrain() {
	_ = p.reader.Close()
	p.drained.Wait()
}

// PreOrigins implements tracer.PreTraceHandler.
func (p *probe) PreOrigins() []uint16 {
	return []uint16{p.launchOrigin}
}

// PreHandleTrace implements tracer.PreTraceHandler. The trace value carries
// the launch's correlation id, kept for PostHandleTrace, and its sample weight,
// left as the value to report.
func (p *probe) PreHandleTrace(trace *libpf.EbpfTrace) bool {
	p.launch.pid = trace.PID
	p.launch.correlationID = uint32(trace.Value)
	trace.Value = int64(max(uint64(trace.Value)>>32, 1))
	return true
}

// PostOrigins implements tracer.PostTraceHandler.
func (p *probe) PostOrigins() []uint16 {
	return []uint16{p.launchOrigin}
}

// PostHandleTrace implements tracer.PostTraceHandler.
func (p *probe) PostHandleTrace(trace *libpf.Trace) {
	p.matcher.addLaunch(p.launch.pid, p.launch.correlationID, trace, time.Now())
}

// ProduceSnapshots implements tracer.SnapshotSource.
func (p *probe) ProduceSnapshots() []samples.SnapshotProfile {
	return p.matcher.snapshot(time.Now())
}

// Unload implements tracer.Probe.
func (p *probe) Unload() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.unloaded = true

	var errs error
	for pid, files := range p.links {
		for _, fileLinks := range files {
			errs = errors.Join(errs, closeLinks(fileLinks))
		}
		delete(p.links, pid)
	}
	p.stopDrain()
	p.closePrograms()
	return errors.Join(errs, p.events.Close())
}

func (p *probe) closePrograms() {
	for _, prog := range p.programs {
		_ = prog.Close()
	}
}

func closeLinks(links []link.Link) error {
	var errs error
	for _, l := range links {
		errs = errors.Join(errs, l.Close())
	}
	return errs
}
