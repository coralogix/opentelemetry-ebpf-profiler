// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

// Package cupti is the profiler side of GPU profiling. It attaches the eBPF
// programs (support/ebpf/gpu_cupti.ebpf.c) to the USDT probes of the
// libotelcupti.so shim and the libotelnccl.so NCCL plugin in each injected
// CUDA process, plus uprobe/uretprobe span pairs on curated cuDNN/cuBLAS
// symbols, and drains four event ringbufs (kernel timing, memcpys, shim
// errors, and misc events: UVM counters, allocations, memsets, stall samples,
// NCCL ops, API spans). A correlation matcher joins GPU-side records to the
// host stack captured at the launch site (on_launch → collect_trace,
// TRACE_GPU, correlation id in the trace value).
package cupti // import "go.opentelemetry.io/ebpf-profiler/gpu/cupti"

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"maps"
	"os"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	cebpf "github.com/cilium/ebpf"
	"github.com/cilium/ebpf/link"
	"github.com/cilium/ebpf/ringbuf"
	"github.com/parca-dev/usdt"

	"go.opentelemetry.io/ebpf-profiler/internal/log"
	"go.opentelemetry.io/ebpf-profiler/libpf/pfunsafe"
)

// shimSoName is the injected CUPTI shim whose USDT probes we attach to.
const shimSoName = "libotelcupti.so"

// expectedArgSpec is the only USDT arg spec the eBPF consumers can decode:
// the single struct pointer pinned to a fixed register by the shim's
// OTELCUPTI_PROBE1_PINNED (see gpu/cupti/usdt_probes.h). A shim built
// differently would make the eBPF read garbage, so attach refuses it.
var expectedArgSpec = map[string]string{
	"amd64": "8@%rax",
	"arm64": "8@x0",
}[runtime.GOARCH]

// KernelTiming mirrors the kernel_executed wire record (otelcupti_kernel_rec
// + pid tail, LE, 56 bytes).
type KernelTiming struct {
	Start         uint64
	End           uint64
	CorrelationID uint32
	DeviceID      uint32
	StreamID      uint32
	GraphID       uint32 // CUDA graph id (0 = not graph-launched)
	NamePtr       uint64 // kernel name in the target process
	NVTXPtr       uint64 // NVTX range nesting at launch (0 = none)
	PID           uint32
	_             uint32 // pad
}

// MemTiming mirrors the gpu_mem wire record (otelcupti_mem_rec + pid tail,
// LE, 56 bytes): one GPU memcpy/memset.
type MemTiming struct {
	Start         uint64
	End           uint64
	CorrelationID uint32
	DeviceID      uint32
	StreamID      uint32
	CopyKind      uint32 // CUpti_ActivityMemcpyKind
	Bytes         uint64
	NVTXPtr       uint64
	PID           uint32
	_             uint32 // pad
}

// seSize is the wire size of GPUShimError, pinned by the _Static_assert in
// gpu_cupti.ebpf.c.
const seSize = 128

// ShimError mirrors struct GPUShimError in gpu_cupti.ebpf.c (LE, 128 bytes):
// a shim-side failure (e.g. another CUPTI subscriber already present).
type ShimError struct {
	Code int32
	PID  uint32
	Msg  string
}

// Event kinds, mirror of enum otelcupti_event_kind in usdt_probes.h.
const (
	EvUVMHtoD   = 1  // V1 = bytes migrated host→device
	EvUVMDtoH   = 2  // V1 = bytes migrated device→host
	EvUVMCPUFlt = 3  // V1 = CPU page faults
	EvUVMGPUFlt = 4  // V1 = GPU page fault groups
	EvAlloc     = 5  // V1 = bytes, V2 = CUpti memoryKind
	EvFree      = 6  // V1 = bytes, V2 = CUpti memoryKind
	EvMemset    = 7  // V1 = bytes, Start/End = GPU ns
	EvStall     = 8  // NamePtr = kernel func, V1 = stall reason ptr, V2 = samples
	EvNCCLOp    = 10 // NamePtr = op name, V1 = traffic bytes
	EvAPISpan   = 11 // V1 = API symbol index (see apiSymbols)
)

// GPUEvent mirrors the gpu_event wire record (otelcupti_event_rec + pid
// tail, LE, 56 bytes).
type GPUEvent struct {
	Start         uint64
	End           uint64
	Kind          uint32
	CorrelationID uint32
	V1            uint64
	V2            uint64
	NamePtr       uint64
	PID           uint32
	_             uint32 // pad
}

// Config wires the loaded eBPF collection (by name) and the event callbacks.
type Config struct {
	Progs    map[string]*cebpf.Program // tracer.EBPFPrograms()
	Maps     map[string]*cebpf.Map     // tracer.EBPFMaps()
	OnTiming func(KernelTiming)
	OnMem    func(MemTiming)
	OnError  func(ShimError)
	OnEvent  func(GPUEvent)
	// OnPIDGone is invoked when an attached PID turns out to be dead (PID
	// reuse detected via starttime) so per-PID aggregation state can be
	// dropped before the new process's events arrive. Optional.
	OnPIDGone func(pid int)
}

// ringbufs the Source drains; validated in New, opened in Start.
var ringbufs = []string{"cupti_events", "cupti_mem_events", "cupti_errors",
	"cupti_misc_events"}

// USDT probe name → eBPF program name. The same programs are attached to the
// shim; the NCCL plugin only carries gpu_event.
var probeProgs = map[string]string{
	"on_launch":       "otel_cupti_on_launch",
	"kernel_executed": "otel_cupti_kernel_executed",
	"gpu_mem":         "otel_cupti_gpu_mem",
	"error":           "otel_cupti_error",
	"gpu_event":       "otel_cupti_gpu_event",
}

// Rescan throttles: a shimless (or attach-failed) PID's maps are re-scanned
// with backoff capped low enough that a late cuInit is picked up quickly;
// attached PIDs are re-scanned for late-dlopen'd libraries (cuDNN, NCCL
// plugin) every extrasRecheck, forever.
const (
	noShimRecheckMin = 10 * time.Second
	noShimRecheckMax = 30 * time.Second
	extrasRecheck    = 10 * time.Second
)

// pidAttachment tracks one process's uprobe links and which instrumentable
// libraries were already attached (more can appear later via dlopen).
type pidAttachment struct {
	links      []link.Link
	libs       map[string]bool
	starttime  uint64 // /proc/<pid>/stat starttime, guards against PID reuse
	nextExtras time.Time
}

type noShimEntry struct {
	next      time.Time
	interval  time.Duration
	starttime uint64 // owner of this backoff; a reused PID must not inherit it
}

// Source attaches to and drains the GPU profiling pipeline. Attachment is
// serialized under mu — it happens at most every rescan tick and takes
// milliseconds, so the simple locking wins over concurrency.
type Source struct {
	cfg     Config
	mu      sync.Mutex
	closed  bool
	pids    map[int]*pidAttachment
	noShim  map[int]noShimEntry
	readers []*ringbuf.Reader
	wg      sync.WaitGroup
}

func New(cfg Config) (*Source, error) {
	if cfg.OnTiming == nil || cfg.OnMem == nil || cfg.OnError == nil || cfg.OnEvent == nil {
		return nil, errors.New(
			"gpu/cupti: OnTiming, OnMem, OnError and OnEvent are required (OnPIDGone is optional)")
	}
	// Missing programs/maps mean the tracer blob was built without
	// gpu_cupti.ebpf.c, LoadGPU is off, or the GPU program load failed
	// non-fatally (e.g. kernel < 5.15 lacks bpf_get_attach_cookie — see the
	// tracer's warnings).
	required := append(slices.Collect(maps.Values(probeProgs)),
		"otel_cupti_api_enter", "otel_cupti_api_exit")
	for _, name := range required {
		if cfg.Progs[name] == nil {
			return nil, fmt.Errorf(
				"gpu/cupti: program %s not loaded (LoadGPU off or load failed, see tracer logs)", name)
		}
	}
	for _, rb := range ringbufs {
		if cfg.Maps[rb] == nil {
			return nil, fmt.Errorf(
				"gpu/cupti: map %s not loaded (LoadGPU off or load failed, see tracer logs)", rb)
		}
	}
	if expectedArgSpec == "" {
		return nil, fmt.Errorf("gpu/cupti: unsupported architecture %s", runtime.GOARCH)
	}
	return &Source{
		cfg:    cfg,
		pids:   make(map[int]*pidAttachment),
		noShim: make(map[int]noShimEntry),
	}, nil
}

// Start begins draining the event ringbufs. Attach per-PID via OnNewPID.
func (s *Source) Start() error {
	handlers := map[string]func([]byte){
		"cupti_events":      s.handleTiming,
		"cupti_mem_events":  s.handleMem,
		"cupti_errors":      s.handleError,
		"cupti_misc_events": s.handleEvent,
	}
	for _, name := range ringbufs {
		r, err := ringbuf.NewReader(s.cfg.Maps[name])
		if err != nil {
			s.closeReaders()
			return fmt.Errorf("%s reader: %w", name, err)
		}
		s.mu.Lock()
		s.readers = append(s.readers, r)
		s.mu.Unlock()
		s.wg.Add(1)
		go s.drain(r, handlers[name])
	}
	return nil
}

func (s *Source) closeReaders() {
	s.mu.Lock()
	rs := s.readers
	s.readers = nil
	s.mu.Unlock()
	for _, r := range rs {
		_ = r.Close()
	}
}

// drain forwards ringbuf records to handle until the reader is closed.
func (s *Source) drain(r *ringbuf.Reader, handle func([]byte)) {
	defer s.wg.Done()
	var rec ringbuf.Record
	errLogged := false
	for {
		if err := r.ReadInto(&rec); err != nil {
			if errors.Is(err, ringbuf.ErrClosed) {
				return
			}
			// Persistent errors (e.g. the map torn down underneath the
			// reader) must not hot-spin.
			if !errLogged {
				errLogged = true
				log.Warnf("gpu/cupti: ringbuf read: %v", err)
			}
			time.Sleep(10 * time.Millisecond)
			continue
		}
		handle(rec.RawSample)
	}
}

// The wire structs mirror the C ringbuf records field-for-field (layouts
// pinned by tests) and both supported architectures are little-endian, so a
// direct cast replaces per-field decoding on this hot path (100k+ events/s).

func (s *Source) handleTiming(b []byte) {
	if t, ok := pfunsafe.Read[KernelTiming](b); ok {
		s.cfg.OnTiming(t)
	}
}

func (s *Source) handleMem(b []byte) {
	if t, ok := pfunsafe.Read[MemTiming](b); ok {
		s.cfg.OnMem(t)
	}
}

func (s *Source) handleError(b []byte) {
	if len(b) < seSize {
		return
	}
	le := binary.LittleEndian
	msg := b[8:seSize]
	if i := bytes.IndexByte(msg, 0); i >= 0 {
		msg = msg[:i]
	}
	s.cfg.OnError(ShimError{
		Code: int32(le.Uint32(b[0:])),
		PID:  le.Uint32(b[4:]),
		Msg:  string(msg),
	})
}

func (s *Source) handleEvent(b []byte) {
	if e, ok := pfunsafe.Read[GPUEvent](b); ok {
		s.cfg.OnEvent(e)
	}
}

// OnNewPID attaches probes to a process if it maps instrumentable libraries
// (CUPTI shim required; NCCL plugin and cuDNN/cuBLAS may be dlopen'd later
// and are re-checked every extrasRecheck for the attachment's lifetime).
// Detects PID reuse via starttime in both the attached and backed-off states,
// so a reused PID neither keeps the dead owner's probes nor inherits its
// backoff. Idempotent; serialized under s.mu.
func (s *Source) OnNewPID(pid int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return
	}
	now := time.Now()
	if pa, ok := s.pids[pid]; ok {
		// A starttime of 0 means /proc was unreadable (here or at attach):
		// identity is unknown, so do NOT detach — a transient read failure
		// must not be mistaken for reuse. Reconcile handles truly-dead PIDs.
		cur := readStarttime(pid)
		if cur == 0 || pa.starttime == 0 || cur == pa.starttime {
			if now.After(pa.nextExtras) {
				pa.nextExtras = now.Add(extrasRecheck)
				s.attachExtrasLocked(pid, pa)
			}
			return
		}
		// PID reuse: the attachment belongs to a dead process. Detach, drop
		// its aggregation state, and fall through to a fresh scan.
		closeLinks(pa.links)
		delete(s.pids, pid)
		if s.cfg.OnPIDGone != nil {
			s.cfg.OnPIDGone(pid)
		}
	}
	if e, ok := s.noShim[pid]; ok && now.Before(e.next) {
		if cur := readStarttime(pid); cur == 0 || e.starttime == 0 || cur == e.starttime {
			return // same (or indeterminable) owner: honor the backoff
		}
		delete(s.noShim, pid) // backoff belonged to the PID's previous owner
	}
	s.scanAndAttachLocked(pid, now)
}

// backoffLocked schedules the next scan for a PID without the shim (or whose
// attach failed). Effective schedule: 10s, 20s, then every 30s.
func (s *Source) backoffLocked(pid int, now time.Time) {
	e := s.noShim[pid]
	if e.starttime == 0 {
		e.starttime = readStarttime(pid)
	}
	e.interval = min(max(2*e.interval, noShimRecheckMin), noShimRecheckMax)
	e.next = now.Add(e.interval)
	s.noShim[pid] = e
}

// scanAndAttachLocked scans the PID's mappings and attaches the shim probes
// plus any instrumentable extras.
func (s *Source) scanAndAttachLocked(pid int, now time.Time) {
	libs, err := scanLibs(pid)
	if err != nil {
		// Unreadable maps (EMFILE, racing exit) is not "no shim": skip the
		// backoff escalation and let the next rescan tick retry.
		return
	}
	shim := libs[shimSoName]
	if shim == "" {
		s.backoffLocked(pid, now)
		return
	}
	links, err := s.attachUSDT(pid, shim, probeProgs)
	if err != nil {
		log.Warnf("gpu/cupti: attach pid %d: %v", pid, err)
		s.backoffLocked(pid, now)
		return
	}
	if len(links) == 0 {
		log.Warnf("gpu/cupti: no otelcupti probes found in %s", shim)
		s.backoffLocked(pid, now)
		return
	}
	pa := &pidAttachment{
		links:      links,
		libs:       map[string]bool{shimSoName: true},
		starttime:  readStarttime(pid),
		nextExtras: now.Add(extrasRecheck),
	}
	s.pids[pid] = pa
	delete(s.noShim, pid)
	log.Infof("gpu/cupti: attached USDT probes to pid %d (%s)", pid, shim)
	s.attachExtrasLocked(pid, pa)
}

// readStarttime returns /proc/<pid>/stat field 22 (process start time in
// clock ticks), or 0 if unreadable.
func readStarttime(pid int) uint64 {
	b, err := os.ReadFile("/proc/" + strconv.Itoa(pid) + "/stat")
	if err != nil {
		return 0
	}
	// Skip past the comm field, which may contain spaces.
	i := bytes.LastIndexByte(b, ')')
	if i < 0 {
		return 0
	}
	fields := strings.Fields(string(b[i+1:]))
	if len(fields) < 20 { // starttime is the 20th field after comm
		return 0
	}
	v, _ := strconv.ParseUint(fields[19], 10, 64)
	return v
}

// attachExtrasLocked attaches the optional per-library probes (NCCL plugin
// USDT, cuDNN/cuBLAS API spans). Failures are not retried; an unreadable
// maps file is retried at the next extras recheck.
func (s *Source) attachExtrasLocked(pid int, pa *pidAttachment) {
	libs, err := scanLibs(pid)
	if err != nil {
		return
	}
	for lib, path := range libs {
		if pa.libs[lib] {
			continue
		}
		pa.libs[lib] = true
		var links []link.Link
		var err error
		if lib == ncclSoName {
			links, err = s.attachUSDT(pid, path, map[string]string{
				"gpu_event": "otel_cupti_gpu_event",
			})
		} else {
			links, err = s.attachAPISymbols(pid, lib, path)
		}
		if err != nil {
			log.Warnf("gpu/cupti: attach %s pid %d: %v", lib, pid, err)
			continue
		}
		if len(links) > 0 {
			pa.links = append(pa.links, links...)
			log.Infof("gpu/cupti: attached %d probes to pid %d (%s)", len(links), pid, lib)
		}
	}
}

// attachUSDT attaches per-PID uprobes to the otelcupti USDT sites in path.
func (s *Source) attachUSDT(pid int, path string,
	want map[string]string) ([]link.Link, error) {
	all, err := usdt.ParseProbesFromFile(path)
	if err != nil {
		return nil, fmt.Errorf("parse USDT: %w", err)
	}
	exe, err := openExecutable(path)
	if err != nil {
		return nil, fmt.Errorf("open executable: %w", err)
	}
	var links []link.Link
	for _, p := range all {
		progName, ok := want[p.Name]
		if !ok {
			continue
		}
		prog := s.cfg.Progs[progName]
		// The library may be a different build than this agent (injected
		// independently); a probe whose arg spec is not the pinned register
		// would be silently misdecoded — refuse it loudly.
		if p.Arguments != expectedArgSpec {
			closeLinks(links)
			return nil, fmt.Errorf("probe %s arg spec %q != %q (build mismatch)",
				p.Name, p.Arguments, expectedArgSpec)
		}
		l, err := exe.Uprobe(p.Name, prog, &link.UprobeOptions{
			Address:      p.Location,
			RefCtrOffset: p.SemaphoreOffset,
			PID:          pid,
		})
		if err != nil {
			closeLinks(links)
			return nil, fmt.Errorf("attach %s at %#x: %w", p.Name, p.Location, err)
		}
		links = append(links, l)
	}
	return links, nil
}

// openExecutable wraps link.OpenExecutable with a chmod fallback: cilium
// v0.21 still requires the exec bit, but pip-installed CUDA libraries ship
// 0644 (uprobes must target the original inode, so copying is no option).
// Drop once a cilium release without the check is pinned.
func openExecutable(path string) (*link.Executable, error) {
	exe, err := link.OpenExecutable(path)
	if err == nil {
		return exe, nil
	}
	if st, serr := os.Stat(path); serr == nil && st.Mode()&0o111 == 0 {
		if cerr := os.Chmod(path, st.Mode()|0o111); cerr == nil {
			return link.OpenExecutable(path)
		}
	}
	return nil, err
}

// attachAPISymbols attaches enter/exit span probes to the curated API symbols
// of a cuDNN/cuBLAS library. Missing symbols are skipped (library versions
// differ); the attach cookie is the symbol's index into apiSymbols.
func (s *Source) attachAPISymbols(pid int, lib, path string) ([]link.Link, error) {
	exe, err := openExecutable(path)
	if err != nil {
		return nil, fmt.Errorf("open executable: %w", err)
	}
	var links []link.Link
	for idx, sym := range apiSymbols {
		if sym.lib != lib {
			continue
		}
		cookie := uint64(idx)
		enter, err := exe.Uprobe(sym.symbol, s.cfg.Progs["otel_cupti_api_enter"],
			&link.UprobeOptions{PID: pid, Cookie: cookie})
		if err != nil {
			continue // symbol not exported by this version
		}
		exit, err := exe.Uretprobe(sym.symbol, s.cfg.Progs["otel_cupti_api_exit"],
			&link.UprobeOptions{PID: pid, Cookie: cookie})
		if err != nil {
			_ = enter.Close()
			continue
		}
		links = append(links, enter, exit)
	}
	return links, nil
}

func closeLinks(links []link.Link) {
	for _, l := range links {
		_ = l.Close()
	}
}

// Reconcile detaches probes from PIDs no longer alive and returns them (so
// the caller can drop their aggregation state). Without this the uprobe fds
// leak and a reused PID would inherit a stale attachment.
func (s *Source) Reconcile(live map[int]struct{}) []int {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil
	}
	var dead []int
	var toClose [][]link.Link
	for pid, pa := range s.pids {
		if _, ok := live[pid]; !ok {
			dead = append(dead, pid)
			toClose = append(toClose, pa.links)
			delete(s.pids, pid)
		}
	}
	for pid := range s.noShim {
		if _, ok := live[pid]; !ok {
			delete(s.noShim, pid)
		}
	}
	s.mu.Unlock()

	for _, links := range toClose {
		closeLinks(links)
	}
	for _, pid := range dead {
		log.Debugf("gpu/cupti: detached probes from exited pid %d", pid)
	}
	return dead
}

// Stop detaches everything and stops the drainers. Safe to call repeatedly.
func (s *Source) Stop() error {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil
	}
	s.closed = true
	rs := s.readers
	pids := s.pids
	s.readers = nil
	s.pids = make(map[int]*pidAttachment)
	s.mu.Unlock()

	for _, r := range rs {
		_ = r.Close()
	}
	for _, pa := range pids {
		closeLinks(pa.links)
	}
	s.wg.Wait()
	return nil
}

// instrumentable library keys; values of scanLibs.
const ncclSoName = "libotelnccl.so"

var apiLibs = []string{"libcudnn.so", "libcublas.so", "libcublasLt.so"}

// scanLibs returns the paths of instrumentable libraries mapped by pid,
// keyed by library name (shim, NCCL plugin, API libraries). Paths are
// resolved through /proc/<pid>/root so they reference the file in the
// process's mount namespace — a same-named host path could be a different
// inode, and uprobes are inode-bound. The error return distinguishes
// "maps unreadable" from "no instrumentable libraries": callers must not
// turn a transient read failure into backoff or detach decisions.
func scanLibs(pid int) (map[string]string, error) {
	out := map[string]string{}
	f, err := os.Open("/proc/" + strconv.Itoa(pid) + "/maps")
	if err != nil {
		return nil, err
	}
	defer f.Close()
	procRoot := "/proc/" + strconv.Itoa(pid) + "/root"
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := sc.Bytes() // byte view: no per-line string alloc
		// Path starts at the first '/' (earlier columns never contain one)
		// and may itself contain spaces, so don't field-split it.
		i := bytes.IndexByte(line, '/')
		if i < 0 {
			continue
		}
		p := bytes.TrimSuffix(line[i:], []byte(" (deleted)"))
		base := p[bytes.LastIndexByte(p, '/')+1:]
		var key string
		switch {
		case bytes.Equal(base, []byte(shimSoName)):
			key = shimSoName
		case bytes.Equal(base, []byte(ncclSoName)):
			key = ncclSoName
		default:
			for _, l := range apiLibs {
				if bytes.HasPrefix(base, []byte(l)) {
					key = l
					break
				}
			}
		}
		if key == "" || out[key] != "" {
			continue
		}
		path := procRoot + string(p)
		if _, err := os.Stat(path); err == nil {
			out[key] = path
		}
	}
	// A mid-stream scan failure means partial results: the shim could be in
	// the unread remainder, so it must not be reported as "not mapped".
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return out, nil
}
