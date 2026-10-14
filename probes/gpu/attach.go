// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package gpu // import "go.opentelemetry.io/ebpf-profiler/probes/gpu"

import (
	"fmt"
	"runtime"
	"slices"

	"github.com/cilium/ebpf/link"

	"go.opentelemetry.io/ebpf-profiler/internal/log"
	"go.opentelemetry.io/ebpf-profiler/libpf"
	"go.opentelemetry.io/ebpf-profiler/libpf/pfelf"
	"go.opentelemetry.io/ebpf-profiler/process"
	"go.opentelemetry.io/ebpf-profiler/usdt"
	"go.opentelemetry.io/ebpf-profiler/util"
)

// usdtProvider is the provider of the otel_gpu USDT contract.
const usdtProvider = "otel_gpu"

// recordArgument is the argument spec of every otel_gpu probe: one pointer,
// pinned to a register.
var recordArgument = map[string]string{"amd64": "8@%rax", "arm64": "8@x0"}[runtime.GOARCH]

// Match implements processmanager.ProbeAttacher. Producers can have any file
// name, so every executable mapping is checked for otel_gpu probes in Attach;
// the USDT discovery results are cached per file.
func (p *probe) Match(process.Process, *process.RawMapping) bool {
	return true
}

// Attach implements processmanager.ProbeAttacher.
func (p *probe) Attach(pr process.Process, mapping *process.RawMapping) error {
	points := p.discover(pr, mapping)
	if len(points) == 0 {
		return nil
	}

	pid := pr.PID()
	file := mapping.GetOnDiskFileIdentifier()

	p.mu.Lock()
	defer p.mu.Unlock()
	if p.unloaded {
		return nil
	}
	if _, ok := p.links[pid][file]; ok {
		return nil
	}

	links, err := p.attachMapping(pr, mapping, points)
	if err != nil {
		return err
	}
	if p.links[pid] == nil {
		p.links[pid] = make(map[util.OnDiskFileIdentifier][]link.Link)
	}
	p.links[pid][file] = links
	log.Debugf("gpu: attached %d probes in %s to PID %d", len(links), mapping.Path, pid)
	return nil
}

// Detach implements processmanager.ProbeAttacher.
func (p *probe) Detach(pid libpf.PID) {
	p.mu.Lock()
	files := p.links[pid]
	delete(p.links, pid)
	p.mu.Unlock()

	for _, links := range files {
		if err := closeLinks(links); err != nil {
			log.Debugf("gpu: detaching PID %d: %v", pid, err)
		}
	}
}

// discover returns the otel_gpu probes of the mapped file that this probe
// consumes.
func (p *probe) discover(pr process.Process, mapping *process.RawMapping) []usdt.AttachmentPoint {
	ref := pfelf.NewReferenceWithOpenFunc(mapping.Path, pr, func() (*pfelf.File, error) {
		f, err := pr.OpenMappingFile(mapping)
		if err != nil {
			return nil, err
		}
		return pfelf.NewFileFromFS(f)
	})
	defer ref.Close()

	points, err := p.discoverer.Discover(ref, mapping.GetOnDiskFileIdentifier())
	var consumed []usdt.AttachmentPoint
	for _, point := range points {
		if point.Provider != usdtProvider {
			continue
		}
		if _, ok := usdtPrograms[point.Name]; ok {
			consumed = append(consumed, point)
		}
	}
	if err != nil && len(consumed) > 0 {
		log.Warnf("gpu: some USDT probes of %s are unusable: %v", mapping.Path, err)
	}
	return consumed
}

// attachMapping attaches the consuming eBPF program to each USDT point,
// restricted to the process. The caller must hold p.mu.
func (p *probe) attachMapping(pr process.Process, mapping *process.RawMapping,
	points []usdt.AttachmentPoint,
) ([]link.Link, error) {
	for _, point := range points {
		if err := checkArguments(point); err != nil {
			return nil, err
		}
	}

	f, err := pr.OpenMappingFile(mapping)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", mapping.Path, err)
	}
	defer f.Close()
	fd, ok := f.(interface{ Fd() uintptr })
	if !ok {
		return nil, fmt.Errorf("%s has no file descriptor", mapping.Path)
	}
	exe, err := link.OpenExecutable(fmt.Sprintf("/proc/self/fd/%d", fd.Fd()))
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", mapping.Path, err)
	}

	// ready goes last: producers profile once its semaphore is raised, so every
	// other probe must work by then. A semaphore rises when its uprobe is
	// installed, before the program is attached.
	readyLast := func(point usdt.AttachmentPoint) int {
		if point.Name == "ready" {
			return 1
		}
		return 0
	}
	slices.SortStableFunc(points, func(a, b usdt.AttachmentPoint) int {
		return readyLast(a) - readyLast(b)
	})

	links := make([]link.Link, 0, len(points))
	for _, point := range points {
		l, err := exe.Uprobe(point.Name, p.programs[usdtPrograms[point.Name]], &link.UprobeOptions{
			Address:      point.Location,
			RefCtrOffset: point.SemaphoreOffset,
			PID:          int(pr.PID()),
		})
		if err != nil {
			_ = closeLinks(links)
			return nil, fmt.Errorf("attach %s:%s in %s: %w",
				point.Provider, point.Name, mapping.Path, err)
		}
		links = append(links, l)
	}
	return links, nil
}

// checkArguments verifies that a probe passes its record pointer where the eBPF
// programs read it; any other placement would be misread silently.
func checkArguments(point usdt.AttachmentPoint) error {
	if point.Arguments == recordArgument {
		return nil
	}
	return fmt.Errorf("probe %s:%s has arguments %q, the otel_gpu contract requires %q",
		point.Provider, point.Name, point.Arguments, recordArgument)
}
