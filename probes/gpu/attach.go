// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package gpu // import "go.opentelemetry.io/ebpf-profiler/probes/gpu"

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"sync"
	"time"

	"github.com/cilium/ebpf/link"
	"golang.org/x/sys/unix"

	"go.opentelemetry.io/ebpf-profiler/internal/log"
	"go.opentelemetry.io/ebpf-profiler/libpf"
	"go.opentelemetry.io/ebpf-profiler/libpf/pfelf"
	"go.opentelemetry.io/ebpf-profiler/process"
	"go.opentelemetry.io/ebpf-profiler/usdt"
	"go.opentelemetry.io/ebpf-profiler/util"
)

// usdtProvider is the provider of the otel_gpu_v1 USDT contract.
const usdtProvider = "otel_gpu_v1"

// unusedFileTimeout is how long the uprobes of a file outlive the last process
// that mapped it, so that an application that restarts finds them in place
// rather than waiting seconds for them to be removed and attached again.
const unusedFileTimeout = time.Minute

// eventArgument is the argument spec of the otel_gpu_v1 probe: one pointer,
// pinned to a register.
var eventArgument = map[string]string{"amd64": "8@%rax", "arm64": "8@x0"}[runtime.GOARCH]

// Match implements processmanager.ProbeAttacher. Producers can have any file
// name, so every executable mapping is checked for the otel_gpu_v1 probe in Attach;
// the USDT discovery results are cached per file.
func (p *probe) Match(process.Process, *process.RawMapping) bool {
	return true
}

// Attach implements processmanager.ProbeAttacher. The uprobes of a file are
// attached once, for every process: the kernel then places them in each process
// that maps the file, from the moment it does. Attaching and removing uprobes
// for each process instead would make a new process wait seconds for the
// processes exiting before it, since the kernel removes uprobes one at a time.
func (p *probe) Attach(pr process.Process, mapping *process.RawMapping, fileID libpf.FileID,
	elfRef *pfelf.Reference,
) error {
	pid := pr.PID()
	file := mapping.GetOnDiskFileIdentifier()
	lastModified := pr.GetMappingFileLastModified(mapping)

	p.mu.Lock()
	defer p.mu.Unlock()
	if p.unloaded {
		return nil
	}
	probes, attached := p.files[file]
	if attached && probes.lastModified == lastModified {
		p.addUser(pid, file)
		return nil
	}
	if refusedAt, ok := p.refused[file]; ok && refusedAt == lastModified {
		return nil
	}

	points := p.discover(elfRef, fileID, mapping.Path)
	if len(points) == 0 {
		return nil
	}
	if err := checkPoints(points); err != nil {
		p.refuseLocked(file, lastModified, mapping.Path, err)
		return nil
	}
	users, pinned := 0, false
	if attached {
		// The file was rewritten in place: its uprobes point into the old code.
		p.removeLocked(file, probes)
		users, pinned = probes.users, probes.pinned
	}
	// Uprobes still being removed would fire next to the new ones.
	p.waitRemovedLocked(file)
	links, err := p.attachMapping(pr, mapping, points)
	if err != nil {
		return err
	}
	p.files[file] = &fileProbes{links: links, lastModified: lastModified, users: users,
		pinned: pinned}
	p.addUser(pid, file)
	log.Debugf("gpu: attached %d probes in %s", len(links), mapping.Path)
	return nil
}

// attachPath attaches to the producer library at path, for every process that
// will load it, and keeps its uprobes while the probe is loaded.
func (p *probe) attachPath(path string) error {
	path, err := filepath.Abs(path)
	if err != nil {
		return err
	}
	var st unix.Stat_t
	if err = unix.Stat(path, &st); err != nil {
		return err
	}
	file := util.OnDiskFileIdentifier{DeviceID: st.Dev, InodeNum: st.Ino}
	fileID, err := libpf.FileIDFromExecutableFile(path)
	if err != nil {
		return err
	}
	ref := pfelf.NewReference(pfelf.FSPath(path), os.DirFS("/"))
	defer ref.Close()

	p.mu.Lock()
	defer p.mu.Unlock()
	if p.unloaded {
		return nil
	}
	if probes, ok := p.files[file]; ok {
		probes.pinned = true
		return nil
	}
	points := p.discover(ref, fileID, path)
	if len(points) == 0 {
		return errors.New("it has no otel_gpu_v1:event probe")
	}
	if err = checkPoints(points); err != nil {
		return err
	}
	exe, err := link.OpenExecutable(path)
	if err != nil {
		return err
	}
	links, err := p.attachPoints(exe, points, path)
	if err != nil {
		return err
	}
	p.files[file] = &fileProbes{links: links, lastModified: st.Mtim.Nano(), pinned: true}
	log.Infof("gpu: attached to producer %s", path)
	return nil
}

// refuseLocked refuses a file whose probe breaks the contract, until it
// changes, and logs why once. The caller must hold p.mu.
func (p *probe) refuseLocked(file util.OnDiskFileIdentifier, lastModified int64, path string,
	err error,
) {
	p.refused[file] = lastModified
	log.Warnf("gpu: refusing %s: %v", path, err)
}

// addUser records that the process maps a file with uprobes. The caller must
// hold p.mu.
func (p *probe) addUser(pid libpf.PID, file util.OnDiskFileIdentifier) {
	if p.users[pid] == nil {
		p.users[pid] = make(libpf.Set[util.OnDiskFileIdentifier])
		p.matcher.processStarted(pid)
	}
	if _, ok := p.users[pid][file]; ok {
		return
	}
	p.users[pid][file] = libpf.Void{}
	probes := p.files[file]
	probes.users++
	if probes.idle != nil {
		probes.idle.Stop()
	}
}

// Detach implements processmanager.ProbeAttacher. The uprobes of the files
// that no process maps any more are removed after unusedFileTimeout.
func (p *probe) Detach(pid libpf.PID) {
	p.mu.Lock()
	for file := range p.users[pid] {
		probes, ok := p.files[file]
		if !ok {
			continue
		}
		probes.users--
		if probes.users > 0 || probes.pinned {
			continue
		}
		if probes.idle == nil {
			probes.idle = time.AfterFunc(unusedFileTimeout, func() {
				p.removeUnused(file, probes)
			})
		} else {
			probes.idle.Reset(unusedFileTimeout)
		}
	}
	delete(p.users, pid)
	p.mu.Unlock()

	p.matcher.processExited(pid, time.Now())
}

// removeUnused removes the uprobes of a file if no process mapped it since its
// last one exited.
func (p *probe) removeUnused(file util.OnDiskFileIdentifier, probes *fileProbes) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.unloaded || p.files[file] != probes || probes.users > 0 || probes.pinned {
		return
	}
	p.removeLocked(file, probes)
}

// removeLocked removes the uprobes of a file in the background: it takes
// seconds, and must not hold up Attach and Detach. The caller must hold p.mu.
func (p *probe) removeLocked(file util.OnDiskFileIdentifier, probes *fileProbes) {
	delete(p.files, file)
	if probes.idle != nil {
		probes.idle.Stop()
	}
	for f, removed := range p.removing {
		select {
		case <-removed:
			delete(p.removing, f)
		default:
		}
	}
	removed := make(chan struct{})
	p.removing[file] = removed
	p.detaching.Go(func() {
		defer close(removed)
		if err := closeLinks(probes.links); err != nil {
			log.Debugf("gpu: removing uprobes: %v", err)
		}
	})
}

// waitRemovedLocked waits until the uprobes of the file, if being removed, are
// gone. Removal does not take p.mu. The caller must hold p.mu.
func (p *probe) waitRemovedLocked(file util.OnDiskFileIdentifier) {
	if removed, ok := p.removing[file]; ok {
		<-removed
		delete(p.removing, file)
	}
}

// discover returns the places of the file at path that fire the otel_gpu_v1
// probe.
func (p *probe) discover(ref *pfelf.Reference, fileID libpf.FileID,
	path string,
) []usdt.AttachmentPoint {
	points, err := p.discoverer.Discover(ref, fileID)
	var consumed []usdt.AttachmentPoint
	for _, point := range points {
		if point.Provider == usdtProvider && point.Name == usdtProbe {
			consumed = append(consumed, point)
		}
	}
	if err != nil && len(consumed) > 0 {
		log.Warnf("gpu: some USDT probes of %s are unusable: %v", path, err)
	}
	return consumed
}

// attachMapping attaches the consuming eBPF program to each USDT point, for
// every process that maps the file. The caller must hold p.mu.
func (p *probe) attachMapping(pr process.Process, mapping *process.RawMapping,
	points []usdt.AttachmentPoint,
) ([]link.Link, error) {
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

	return p.attachPoints(exe, points, mapping.Path)
}

// attachPoints attaches the program to each place firing the probe, all at
// once: installing a uprobe waits for the kernel, about a quarter second each on
// recent kernels.
func (p *probe) attachPoints(exe *link.Executable, points []usdt.AttachmentPoint,
	path string,
) ([]link.Link, error) {
	links := make([]link.Link, len(points))
	errs := make([]error, len(points))
	var wg sync.WaitGroup
	for i, point := range points {
		wg.Go(func() {
			links[i], errs[i] = exe.Uprobe(point.Name, p.program,
				&link.UprobeOptions{
					Address:      point.Location,
					RefCtrOffset: point.SemaphoreOffset,
				})
			if errs[i] != nil {
				errs[i] = fmt.Errorf("attach %s:%s in %s: %w", point.Provider, point.Name, path,
					errs[i])
			}
		})
	}
	wg.Wait()
	if err := errors.Join(errs...); err != nil {
		_ = closeLinks(slices.DeleteFunc(links, func(l link.Link) bool { return l == nil }))
		return nil, err
	}
	return links, nil
}

// checkPoints verifies that a file fires the probe from a single place, and
// passes the event pointer where the eBPF program reads it.
func checkPoints(points []usdt.AttachmentPoint) error {
	if len(points) > 1 {
		return fmt.Errorf("it fires otel_gpu_v1:event from %d places, the contract allows one",
			len(points))
	}
	for _, point := range points {
		if err := checkArguments(point); err != nil {
			return err
		}
	}
	return nil
}

// checkArguments verifies that a probe passes its event pointer where the eBPF
// program reads it; any other placement would be misread silently.
func checkArguments(point usdt.AttachmentPoint) error {
	if point.Arguments == eventArgument {
		return nil
	}
	return fmt.Errorf("probe %s:%s has arguments %q, the otel_gpu_v1 contract requires %q",
		point.Provider, point.Name, point.Arguments, eventArgument)
}
