// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

// Package gpu implements GPU profiling as a probe. Workloads emit the otel_gpu
// USDT probes defined in otel_gpu_usdt.h through an injected producer, such as
// a CUPTI library loaded with CUDA_INJECTION64_PATH. The probe attaches to
// those probes per process and captures the host stack at each GPU call site.
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

	cebpf "github.com/cilium/ebpf"
	"github.com/cilium/ebpf/link"

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
	"ready":     "uprobe__gpu_ready",
	"on_launch": "uprobe__gpu_launch",
}

type probe struct {
	config Config

	// discoverer is only used from Attach, which the process manager
	// serializes.
	discoverer *usdt.Discoverer

	// programs holds the loaded eBPF programs by name; set once by Load.
	programs map[string]*cebpf.Program

	launchOrigin uint16

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

	programNames := make([]string, 0, len(usdtPrograms))
	loaders := make([]tracer.ProgLoaderHelper, 0, len(usdtPrograms))
	for _, name := range usdtPrograms {
		programNames = append(programNames, name)
		loaders = append(loaders, tracer.ProgLoaderHelper{
			Name: name, NoTailCallTarget: true, Enable: true,
		})
	}
	coll, err := probeCtx.CollectionSpecWith(nil, programNames, []string{"origin_id_gpu_launch"})
	if err != nil {
		return err
	}
	if err := coll.Variables["origin_id_gpu_launch"].Set(launchOrigin); err != nil {
		return fmt.Errorf("set origin_id_gpu_launch: %w", err)
	}
	if err := probeCtx.RewriteMaps(coll, nil); err != nil {
		return err
	}

	programs := make(map[string]*cebpf.Program)
	if err := probeCtx.LoadProbeUnwinders(coll, programs, loaders, 0); err != nil {
		return err
	}
	p.programs = programs

	if err := probeCtx.AddAttacher(p); err != nil {
		p.closePrograms()
		return fmt.Errorf("registering process attacher: %w", err)
	}
	return nil
}

// PreOrigins implements tracer.PreTraceHandler.
func (p *probe) PreOrigins() []uint16 {
	return []uint16{p.launchOrigin}
}

// PreHandleTrace implements tracer.PreTraceHandler. The trace value carries
// the launch's correlation id and its sample weight; the weight is left as the
// value to report.
func (p *probe) PreHandleTrace(trace *libpf.EbpfTrace) bool {
	trace.Value = int64(max(uint64(trace.Value)>>32, 1))
	return true
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
	p.closePrograms()
	return errs
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
