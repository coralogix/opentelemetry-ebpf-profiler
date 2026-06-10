// SPDX-License-Identifier: Apache-2.0

package cupti

import (
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"go.opentelemetry.io/ebpf-profiler/internal/log"
	"go.opentelemetry.io/ebpf-profiler/libpf"
	"go.opentelemetry.io/ebpf-profiler/reporter/samples"
	"go.opentelemetry.io/ebpf-profiler/traceutil"
)

// launchCtx is the (pre-collapsed) host stack captured at a tracked CUDA call
// site, buffered until the matching GPU activity record arrives.
type launchCtx struct {
	mods    []string // leaf-first module names, consecutive dups merged
	pathKey string   // mods joined with '\x00'
	meta    samples.TraceEventMeta
}

// bucketKey identifies one export bucket. Comparable struct so hot-path map
// lookups don't build key strings.
type bucketKey struct {
	origin libpf.Origin
	pid    libpf.PID
	nvtx   string
	name   string
	path   string
}

// emitBucket accumulates a value for one bucket; each flush reports only the
// growth past the emitted watermark. The trace is built once at creation.
type emitBucket struct {
	trace   *libpf.Trace
	meta    samples.TraceEventMeta
	origin  libpf.Origin
	weight  int64
	emitted int64
	idle    int // consecutive zero-delta flushes; evicted at bucketMaxIdle
}

// Matcher joins host stacks from on_launch with GPU activity records, keyed
// by (PID, CUPTI correlation id) — correlation ids are per-process counters.
// Aggregates per (origin, process, NVTX, leaf, host stack); Flush reports the
// deltas. Host-stack capture is sampled; kernel timing itself is complete.
type Matcher struct {
	mu   sync.Mutex
	buf  map[uint64]launchCtx // key: PID<<32 | correlation id
	ring []uint64             // FIFO eviction of buf (oldest first)
	head int
	tp   *NameCache
	rep  TraceReporter
	orig Origins
	emit map[bucketKey]*emitBucket
	comm map[libpf.PID]libpf.String // fallback comm cache

	// flushMu serializes Flush: concurrent flushes would snapshot the same
	// delta and double-report it.
	flushMu sync.Mutex

	matched   uint64
	unmatched uint64
}

// Origins are the dynamic probe origins the matcher reports under; see
// the registration table in Run for the type/unit each one carries.
type Origins struct {
	KernelTime   libpf.Origin
	MemTime      libpf.Origin
	MemBytes     libpf.Origin
	BusyTime     libpf.Origin
	UVMBytes     libpf.Origin
	UVMFaults    libpf.Origin
	AllocBytes   libpf.Origin
	StallSamples libpf.Origin
	NCCLTime     libpf.Origin
	NCCLBytes    libpf.Origin
	APITime      libpf.Origin
}

// launchBufCap bounds the launch buffer; the FIFO ring evicts oldest-first
// (matching is lossy by design — evicted records count as unmatched).
const launchBufCap = 1 << 17

// bucketMaxIdle evicts export buckets after this many consecutive zero-delta
// flushes, bounding emit-map growth for long-lived processes (a bucket is
// rebuilt on demand if its identity becomes active again).
const bucketMaxIdle = 24 // 2 minutes at the 5s flush cadence

func NewMatcher(rep TraceReporter, origins Origins, tp *NameCache) *Matcher {
	return &Matcher{
		buf: make(map[uint64]launchCtx),
		// ring is allocated lazily on the first launch: an idle agent with
		// --gpu should not pin ~4 MB of map/ring capacity.
		tp:   tp,
		rep:  rep,
		orig: origins,
		emit: make(map[bucketKey]*emitBucket),
		comm: make(map[libpf.PID]libpf.String),
	}
}

// TraceReporter is the sink Flush reports traces to.
type TraceReporter interface {
	ReportTraceEvent(trace *libpf.Trace, meta *samples.TraceEventMeta) error
}

func bufKey(pid libpf.PID, corr uint32) uint64 {
	return uint64(pid)<<32 | uint64(corr)
}

// OnLaunchTrace buffers the host stack for a tracked CUDA call. The
// correlation id is carried in meta.Value (set by otel_cupti_on_launch).
// The stack is collapsed to one frame per shared object up front — stripped
// CUDA stacks have no symbols, so raw frames repeat the same .so name.
func (m *Matcher) OnLaunchTrace(trace *libpf.Trace, meta *samples.TraceEventMeta) {
	corr := uint32(meta.Value)
	if corr == 0 {
		return
	}
	mods := collapsedModules(trace.Frames)
	if len(mods) == 0 {
		return
	}
	lc := launchCtx{mods: mods, pathKey: strings.Join(mods, "\x00"), meta: *meta}
	key := bufKey(meta.PID, corr)
	m.mu.Lock()
	if m.ring == nil {
		m.ring = make([]uint64, launchBufCap)
	}
	if _, ok := m.buf[key]; !ok {
		if old := m.ring[m.head]; old != 0 {
			delete(m.buf, old)
		}
		m.ring[m.head] = key
		m.head = (m.head + 1) % len(m.ring)
	}
	m.buf[key] = lc
	m.mu.Unlock()
}

// bucket returns (creating if needed) the bucket for key. The trace is built
// once: leaf, NVTX nesting (innermost first, '\x1f'-joined by the shim),
// host modules. Caller must hold m.mu.
func (m *Matcher) bucket(key bucketKey, mods []string,
	meta samples.TraceEventMeta) *emitBucket {
	eb := m.emit[key]
	if eb != nil {
		return eb
	}
	var frames libpf.Frames
	frames.Append(&libpf.Frame{Type: libpf.GPUKernelFrame, FunctionName: libpf.Intern(key.name)})
	if key.nvtx != "" {
		for _, r := range strings.Split(key.nvtx, "\x1f") {
			if r != "" {
				frames.Append(&libpf.Frame{Type: libpf.NativeFrame,
					FunctionName: libpf.Intern("nvtx:" + r)})
			}
		}
	}
	for _, mod := range mods {
		frames.Append(&libpf.Frame{Type: libpf.NativeFrame, FunctionName: libpf.Intern(mod)})
	}
	trace := &libpf.Trace{Frames: frames}
	trace.Hash = traceutil.HashTrace(trace)
	eb = &emitBucket{trace: trace, meta: meta, origin: key.origin}
	m.emit[key] = eb
	return eb
}

// OnTiming folds one kernel-timing event into the export buckets.
func (m *Matcher) OnTiming(t KernelTiming) {
	dur := int64(t.End - t.Start)
	if dur <= 0 {
		return
	}
	name := m.tp.resolveName(t.PID, t.NamePtr)
	if name == "" {
		name = "[unknown-kernel]"
	}
	nvtx := ""
	if t.NVTXPtr != 0 {
		nvtx = m.tp.resolveName(t.PID, t.NVTXPtr)
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	key := bufKey(libpf.PID(t.PID), t.CorrelationID)
	bk := bucketKey{origin: m.orig.KernelTime, pid: libpf.PID(t.PID), nvtx: nvtx, name: name}
	var mods []string
	var meta samples.TraceEventMeta
	if lc, ok := m.buf[key]; ok {
		// A CUDA graph launch expands into many kernels sharing one
		// correlation id; keep the entry so all of them attribute to the
		// graph-launch stack.
		if t.GraphID == 0 {
			delete(m.buf, key)
		}
		m.matched++
		mods, bk.path, meta = lc.mods, lc.pathKey, lc.meta
	} else {
		// Keep the timing: kernel time is the authoritative total, so an
		// unmatched record still counts, attributed to the process only.
		m.unmatched++
		meta = m.fallbackMetaLocked(bk.pid)
	}
	m.bucket(bk, mods, meta).weight += dur
}

const copyKindMemset = 100 // synthetic kind: memsets share the memcpy pipeline

// copyKinds indexes CUpti_ActivityMemcpyKind.
var copyKinds = [...]string{1: "HtoD", 2: "DtoH", 3: "HtoA", 4: "AtoH", 5: "AtoA",
	6: "AtoD", 7: "DtoA", 8: "DtoD", 9: "HtoH", 10: "PtoP"}

func copyKindName(kind uint32) string {
	if kind == copyKindMemset {
		return "memset"
	}
	if int(kind) < len(copyKinds) && copyKinds[kind] != "" {
		return "memcpy:" + copyKinds[kind]
	}
	return "memcpy:unknown"
}

// OnMem folds one memcpy/memset event into two buckets of the same identity:
// duration under MemTime and bytes under MemBytes, host-stack-joined when the
// correlation id matches. The buffered entry is NOT evicted: a CUDA graph
// with memcpy nodes shares its id with the graph's kernels.
func (m *Matcher) OnMem(t MemTiming) {
	dur := int64(t.End - t.Start)
	if dur <= 0 {
		return
	}
	name := copyKindName(t.CopyKind)
	pid := libpf.PID(t.PID)
	nvtx := ""
	if t.NVTXPtr != 0 {
		nvtx = m.tp.resolveName(t.PID, t.NVTXPtr)
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	mods, pathKey, meta := m.launchOrFallbackLocked(pid, t.CorrelationID)
	key := bucketKey{origin: m.orig.MemTime, pid: pid, nvtx: nvtx, name: name, path: pathKey}
	m.bucket(key, mods, meta).weight += dur
	if t.Bytes > 0 {
		key.origin = m.orig.MemBytes
		m.bucket(key, mods, meta).weight += int64(t.Bytes)
	}
}

// launchOrFallbackLocked returns the buffered launch stack for (pid, corr)
// or a comm-only fallback identity. Caller must hold m.mu.
func (m *Matcher) launchOrFallbackLocked(pid libpf.PID, corr uint32) (
	mods []string, pathKey string, meta samples.TraceEventMeta) {
	if lc, ok := m.buf[bufKey(pid, corr)]; ok {
		return lc.mods, lc.pathKey, lc.meta
	}
	return nil, "", m.fallbackMetaLocked(pid)
}

// fallbackMetaLocked returns process identity for events with no buffered
// launch stack. Caller must hold m.mu.
func (m *Matcher) fallbackMetaLocked(pid libpf.PID) samples.TraceEventMeta {
	comm, ok := m.comm[pid]
	if !ok {
		comm = readComm(int(pid))
		m.comm[pid] = comm
	}
	return samples.TraceEventMeta{PID: pid, Comm: comm}
}

// AddSample folds value into the bucket for (pid, frames, origin); frames is
// leaf-first. For events without correlation ids (busy time, UVM, stalls,
// NCCL, API spans).
func (m *Matcher) AddSample(pid uint32, frames []string, origin libpf.Origin, value int64) {
	if value <= 0 || len(frames) == 0 {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.bucket(bucketKey{
		origin: origin,
		pid:    libpf.PID(pid),
		name:   frames[0],
		path:   strings.Join(frames[1:], "\x00"),
	}, frames[1:], m.fallbackMetaLocked(libpf.PID(pid))).weight += value
}

// memKinds indexes CUpti_ActivityMemoryKind.
var memKinds = [...]string{1: "pageable", 2: "pinned", 3: "device", 4: "array",
	5: "managed", 6: "device_static", 7: "managed_static"}

func cuptiMemKindName(kind uint64) string {
	if kind < uint64(len(memKinds)) && memKinds[kind] != "" {
		return memKinds[kind]
	}
	return "unknown"
}

// OnEvent dispatches a generic GPU event into the export buckets.
func (m *Matcher) OnEvent(e GPUEvent) {
	switch e.Kind {
	case EvUVMHtoD:
		m.AddSample(e.PID, []string{"uvm:HtoD"}, m.orig.UVMBytes, int64(e.V1))
	case EvUVMDtoH:
		m.AddSample(e.PID, []string{"uvm:DtoH"}, m.orig.UVMBytes, int64(e.V1))
	case EvUVMCPUFlt:
		m.AddSample(e.PID, []string{"uvm:cpu_fault"}, m.orig.UVMFaults, int64(e.V1))
	case EvUVMGPUFlt:
		m.AddSample(e.PID, []string{"uvm:gpu_fault"}, m.orig.UVMFaults, int64(e.V1))
	case EvAlloc, EvFree:
		op := "alloc:"
		if e.Kind == EvFree {
			op = "free:"
		}
		name := op + cuptiMemKindName(e.V2)
		pid := libpf.PID(e.PID)
		// cudaMalloc/cuMemAlloc call sites are tracked, so the correlation id
		// usually joins to a host stack.
		m.mu.Lock()
		mods, pathKey, meta := m.launchOrFallbackLocked(pid, e.CorrelationID)
		m.bucket(bucketKey{
			origin: m.orig.AllocBytes,
			pid:    pid,
			name:   name,
			path:   pathKey,
		}, mods, meta).weight += int64(e.V1)
		m.mu.Unlock()
	case EvMemset:
		m.OnMem(MemTiming{
			Start:         e.Start,
			End:           e.End,
			CorrelationID: e.CorrelationID,
			CopyKind:      copyKindMemset,
			Bytes:         e.V1,
			PID:           e.PID,
		})
	case EvStall:
		fn := m.tp.resolveName(e.PID, e.NamePtr)
		reason := m.tp.resolveName(e.PID, e.V1)
		if fn == "" || reason == "" {
			return
		}
		m.AddSample(e.PID, []string{"stall:" + reason, fn}, m.orig.StallSamples, int64(e.V2))
	case EvNCCLOp:
		op := m.tp.resolveName(e.PID, e.NamePtr)
		if op == "" {
			op = "unknown"
		}
		name := "nccl:" + op
		pid := libpf.PID(e.PID)
		m.mu.Lock()
		meta := m.fallbackMetaLocked(pid)
		if dur := int64(e.End - e.Start); dur > 0 {
			m.bucket(bucketKey{origin: m.orig.NCCLTime, pid: pid, name: name},
				nil, meta).weight += dur
		}
		if e.V1 > 0 {
			m.bucket(bucketKey{origin: m.orig.NCCLBytes, pid: pid, name: name},
				nil, meta).weight += int64(e.V1)
		}
		m.mu.Unlock()
	case EvAPISpan:
		dur := int64(e.End - e.Start)
		// A stale eBPF entry (missed uretprobe) pairs a fresh exit with an
		// ancient start; no real cuDNN/cuBLAS call runs for minutes.
		if dur > int64(time.Minute) {
			return
		}
		m.AddSample(e.PID, []string{apiSymbolDisplay(e.V1)}, m.orig.APITime, dur)
	}
}

// readComm returns the process's comm name, or NullString if unreadable.
func readComm(pid int) libpf.String {
	b, err := os.ReadFile("/proc/" + strconv.Itoa(pid) + "/comm")
	if err != nil {
		return libpf.NullString
	}
	return libpf.Intern(strings.TrimSpace(string(b)))
}

// Flush reports each bucket's growth since the last flush. Call periodically
// and once on shutdown. Serialized: concurrent flushes would report the same
// delta twice.
func (m *Matcher) Flush() {
	if m.rep == nil {
		return
	}
	m.flushMu.Lock()
	defer m.flushMu.Unlock()
	now := libpf.UnixTime64(time.Now().UnixNano())

	type pending struct {
		eb    *emitBucket
		delta int64
	}
	m.mu.Lock()
	out := make([]pending, 0, len(m.emit))
	for key, eb := range m.emit {
		d := eb.weight - eb.emitted
		if d > 0 {
			eb.idle = 0
			out = append(out, pending{eb: eb, delta: d})
		} else if eb.idle++; eb.idle >= bucketMaxIdle {
			delete(m.emit, key)
		}
	}
	buckets, matched, unmatched := len(m.emit), m.matched, m.unmatched
	m.mu.Unlock()
	log.Debugf("gpu/cupti: flush %d/%d buckets (matched %d, unmatched %d)",
		len(out), buckets, matched, unmatched)

	for _, p := range out {
		meta := p.eb.meta
		meta.Origin = p.eb.origin
		meta.Value = p.delta
		meta.Timestamp = now
		if err := m.rep.ReportTraceEvent(p.eb.trace, &meta); err != nil {
			// Watermark stays; the next flush retries this delta.
			log.Debugf("gpu/cupti: report GPU trace: %v", err)
			continue
		}
		// Advance by delta, not to weight — it may have grown concurrently.
		// Advancing a concurrently-pruned bucket is harmless.
		m.mu.Lock()
		p.eb.emitted += p.delta
		m.mu.Unlock()
	}
}

// collapsedModules returns the leaf-first host module (shared-object) names,
// merging consecutive frames from the same module.
func collapsedModules(frames libpf.Frames) []string {
	out := make([]string, 0, 16)
	last := ""
	var lastMapping libpf.FrameMapping
	for _, h := range frames {
		f := h.Value()
		// Consecutive frames overwhelmingly share a mapping; skip them
		// before the (comparatively costly) name extraction.
		if f.Mapping.Valid() && f.Mapping == lastMapping {
			continue
		}
		lastMapping = f.Mapping
		name := ""
		if f.Mapping.Valid() {
			name = f.Mapping.Value().File.Value().FileName.String()
			if i := strings.LastIndexByte(name, '/'); i >= 0 {
				name = name[i+1:]
			}
		}
		if name == "" {
			name = f.FunctionName.String()
			if name == "" {
				continue
			}
		}
		if name == last {
			continue
		}
		out = append(out, name)
		last = name
	}
	return out
}

// PruneDeadPIDs drops export buckets, launch contexts and caches for exited
// PIDs (a reused PID must not inherit the dead process's state).
func (m *Matcher) PruneDeadPIDs(dead []int) {
	if len(dead) == 0 {
		return
	}
	deadSet := make(map[libpf.PID]struct{}, len(dead))
	for _, p := range dead {
		deadSet[libpf.PID(p)] = struct{}{}
		m.tp.ForgetPID(uint32(p))
	}
	m.mu.Lock()
	for k := range m.emit {
		if _, ok := deadSet[k.pid]; ok {
			delete(m.emit, k)
		}
	}
	for k := range m.buf {
		if _, ok := deadSet[libpf.PID(k>>32)]; ok {
			delete(m.buf, k)
		}
	}
	for pid := range deadSet {
		delete(m.comm, pid)
	}
	m.mu.Unlock()
}

// SweepCaches drops comm and name-cache entries whose PID is not in live.
// Late ringbuf events can repopulate them after a prune, and fallback paths
// create entries for PIDs that were never attached.
func (m *Matcher) SweepCaches(live map[int]struct{}) {
	m.mu.Lock()
	for pid := range m.comm {
		if _, ok := live[int(pid)]; !ok {
			delete(m.comm, pid)
		}
	}
	m.mu.Unlock()
	m.tp.Sweep(live)
}

// Stats returns matched / unmatched timing counts.
func (m *Matcher) Stats() (matched, unmatched uint64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.matched, m.unmatched
}
