// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

// Package gpu implements GPU profiling as a probe. Workloads fire the
// otel_gpu_v1:event USDT probe defined in otel_gpu_v1_usdt.h through an injected
// producer, such as a CUPTI library loaded with CUDA_INJECTION64_PATH. The probe
// attaches to it once per producer library, captures the host stack at each GPU
// call and attributes the GPU time measured later to that stack.
//
// An OTel config using this probe could look like this:
//
//	receivers:
//	  profiling:
//	    probes:
//	      - gpu
//
//	extensions:
//	  gpu:
//	    # Optional: producer libraries to attach to at start.
//	    producer_paths: [/opt/otelgpu/libotelcupti.so]
//
//	service:
//	  extensions: [gpu]
package gpu // import "go.opentelemetry.io/ebpf-profiler/probes/gpu"

import (
	"context"
	"errors"
	"fmt"
	"math"
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

// Config is the configuration of the gpu probe, which has no options yet.
type Config struct {
	// ProducerPaths are producer libraries to attach to at start, so that every
	// process loading one is covered from its first GPU call, rather than from
	// when the profiler scans its mappings.
	ProducerPaths []string `mapstructure:"producer_paths"`
}

// launchProfileType counts GPU calls per host stack, each as often as its
// sample weight.
var launchProfileType = &samples.TypeMetadata{
	SampleTypes:  []samples.SampleType{{Type: "gpu_launches", Unit: "count"}},
	ReportValues: true,
	AppendValues: appendLaunchWeight,
}

// Context values of a launch trace, which support/ebpf/gpu.ebpf.c pushes.
const (
	launchCorrelationID = iota
	launchSampleWeight
	launchContextValues
)

// appendLaunchWeight appends the sample weight of a launch; a weight of 0
// counts as 1.
func appendLaunchWeight(dst []int64, meta *samples.TraceEventMeta) []int64 {
	if len(meta.ContextValues) < launchContextValues {
		return append(dst, 1)
	}
	return append(dst, int64(max(meta.ContextValues[launchSampleWeight], 1)))
}

const (
	// usdtProbe is the only probe of the otel_gpu_v1 contract.
	usdtProbe = "event"
	// eventProgram is the eBPF program consuming it.
	eventProgram = "uprobe__gpu_event"
	// eventsMap carries the events from eBPF.
	eventsMap = "gpu_events"
	// scratchMap holds the event eBPF is forwarding.
	scratchMap = "gpu_event_scratch"
)

type probe struct {
	config Config

	// discoverer is only used from Attach, which the process manager
	// serializes.
	discoverer *usdt.Discoverer

	// program consumes the probe; set once by Load.
	program *cebpf.Program

	launchOrigin uint16
	// launch is the launch whose trace is being symbolized. PreHandleTrace
	// sets it for PostHandleTrace, both called from the trace handling loop.
	launch struct {
		pid           libpf.PID
		correlationID uint64
	}

	matcher *matcher
	names   *nameCache
	events  *cebpf.Map
	scratch *cebpf.Map
	reader  *ringbuf.Reader
	drained sync.WaitGroup
	// decodeErrorLogged is set once an undecodable event was logged; it is
	// only used by the goroutine reading events.
	decodeErrorLogged bool

	// mu serializes attaching with Unload, which closes the program.
	mu       sync.Mutex
	unloaded bool
	// files holds the uprobes of each file that has some, and users the files
	// with uprobes that each process maps.
	files map[util.OnDiskFileIdentifier]*fileProbes
	users map[libpf.PID]libpf.Set[util.OnDiskFileIdentifier]
	// removing is closed for each file once the uprobes it had are removed.
	removing map[util.OnDiskFileIdentifier]chan struct{}
	// refused holds the modification time of each file whose probe breaks the
	// contract, so that it is refused and logged once.
	refused map[util.OnDiskFileIdentifier]int64
	// detaching tracks the uprobes still being removed.
	detaching sync.WaitGroup
}

// fileProbes are the uprobes of a file, and the number of processes that map it.
type fileProbes struct {
	links []link.Link
	// lastModified is the modification time of the file the uprobes were
	// placed in.
	lastModified int64
	users        int
	// pinned uprobes, of a producer path of the configuration, stay while the
	// probe is loaded.
	pinned bool
	// idle removes the uprobes once no process has mapped the file for a while.
	idle *time.Timer
}

// New returns the gpu probe.
func New(config Config) (tracer.Probe, error) {
	discoverer, err := usdt.NewDiscoverer()
	if err != nil {
		return nil, err
	}
	names, err := newNameCache()
	if err != nil {
		return nil, err
	}
	return &probe{
		config:     config,
		discoverer: discoverer,
		names:      names,
		files:      make(map[util.OnDiskFileIdentifier]*fileProbes),
		users:      make(map[libpf.PID]libpf.Set[util.OnDiskFileIdentifier]),
		removing:   make(map[util.OnDiskFileIdentifier]chan struct{}),
		refused:    make(map[util.OnDiskFileIdentifier]int64),
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
	if p.matcher, err = newMatcher(probeCtx.ProcessMeta); err != nil {
		return err
	}

	coll, err := probeCtx.CollectionSpecWith(
		[]string{eventsMap, scratchMap},
		[]string{eventProgram},
		[]string{"origin_id_gpu_launch"},
	)
	if err != nil {
		return err
	}
	if err := coll.Variables["origin_id_gpu_launch"].Set(launchOrigin); err != nil {
		return fmt.Errorf("set origin_id_gpu_launch: %w", err)
	}

	if err := p.createMaps(coll); err != nil {
		return err
	}
	maps := map[string]*cebpf.Map{eventsMap: p.events, scratchMap: p.scratch}
	if err := probeCtx.RewriteMaps(coll, maps); err != nil {
		p.closeMaps()
		return err
	}
	programs := make(map[string]*cebpf.Program)
	loaders := []tracer.ProgLoaderHelper{{Name: eventProgram, NoTailCallTarget: true, Enable: true}}
	if err := probeCtx.LoadProbeUnwinders(coll, programs, loaders, 0); err != nil {
		p.closeMaps()
		return err
	}
	p.program = programs[eventProgram]

	if p.reader, err = ringbuf.NewReader(p.events); err != nil {
		p.closeProgram()
		p.closeMaps()
		return fmt.Errorf("reading %s: %w", eventsMap, err)
	}
	p.drained.Add(1)
	go p.drain()

	if err := probeCtx.AddAttacher(p); err != nil {
		p.stopDrain()
		p.closeProgram()
		p.closeMaps()
		return fmt.Errorf("registering process attacher: %w", err)
	}
	for _, path := range p.config.ProducerPaths {
		if err := p.attachPath(path); err != nil {
			log.Warnf("gpu: not attaching to producer %s: %v", path, err)
		}
	}
	return nil
}

// createMaps creates the maps the probe owns.
func (p *probe) createMaps(coll *cebpf.CollectionSpec) error {
	var err error
	if p.events, err = cebpf.NewMap(coll.Maps[eventsMap]); err != nil {
		return fmt.Errorf("creating %s: %w", eventsMap, err)
	}
	if p.scratch, err = cebpf.NewMap(coll.Maps[scratchMap]); err != nil {
		p.closeMaps()
		return fmt.Errorf("creating %s: %w", scratchMap, err)
	}
	return nil
}

func (p *probe) closeMaps() error {
	var errs error
	for _, m := range []*cebpf.Map{p.events, p.scratch} {
		if m != nil {
			errs = errors.Join(errs, m.Close())
		}
	}
	return errs
}

// drain forwards the events from eBPF until the reader is closed.
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
				log.Warnf("gpu: reading GPU events: %v", err)
				errorLogged = true
			}
			// Avoid spinning on a persistent error.
			time.Sleep(10 * time.Millisecond)
			continue
		}
		p.handleEvent(record.RawSample)
	}
}

func (p *probe) handleEvent(b []byte) {
	ev, err := decodeEvent(b)
	if err != nil {
		if !p.decodeErrorLogged {
			log.Warnf("gpu: dropping an event that does not decode: %v", err)
			p.decodeErrorLogged = true
		}
		return
	}
	now := time.Now()
	if ev.kind == kindDevice {
		if ev.gpu != "" && ev.gpuIndex != 0 && ev.gpuIndex <= math.MaxUint32 {
			p.matcher.setGPU(ev.pid, uint32(ev.gpuIndex), ev.gpu, now)
		}
		return
	}
	if ev.kind == kindKernel && ev.name != "" {
		ev.name = p.names.demangle(ev.name)
	}
	if work, ok := eventWork(ev); ok {
		p.matcher.addWork(work, now)
	}
}

func (p *probe) stopDrain() {
	if p.reader != nil {
		_ = p.reader.Close()
	}
	p.drained.Wait()
}

// PreOrigins implements tracer.PreTraceHandler.
func (p *probe) PreOrigins() []uint16 {
	return []uint16{p.launchOrigin}
}

// PreHandleTrace implements tracer.PreTraceHandler. It keeps the launch's
// correlation id for PostHandleTrace.
func (p *probe) PreHandleTrace(trace *libpf.EbpfTrace) bool {
	if len(trace.ContextValues) < launchContextValues {
		return false
	}
	p.launch.pid = trace.PID
	p.launch.correlationID = trace.ContextValues[launchCorrelationID]
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
	for file, probes := range p.files {
		if probes.idle != nil {
			probes.idle.Stop()
		}
		errs = errors.Join(errs, closeLinks(probes.links))
		delete(p.files, file)
	}
	clear(p.users)
	p.detaching.Wait()
	p.stopDrain()
	p.closeProgram()
	return errors.Join(errs, p.closeMaps())
}

func (p *probe) closeProgram() {
	if p.program != nil {
		_ = p.program.Close()
	}
}

// closeLinks closes links all at once: removing a uprobe waits for the kernel,
// about a quarter second each on recent kernels.
func closeLinks(links []link.Link) error {
	errs := make([]error, len(links))
	var wg sync.WaitGroup
	for i, l := range links {
		wg.Go(func() { errs[i] = l.Close() })
	}
	wg.Wait()
	return errors.Join(errs...)
}
