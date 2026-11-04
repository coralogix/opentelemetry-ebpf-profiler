// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package gpu // import "go.opentelemetry.io/ebpf-profiler/probes/gpu"

import (
	"strings"
	"sync"
	"time"

	lru "github.com/elastic/go-freelru"
	"github.com/zeebo/xxh3"
	"go.opentelemetry.io/otel/attribute"

	"go.opentelemetry.io/ebpf-profiler/libpf"
	"go.opentelemetry.io/ebpf-profiler/reporter/samples"
)

const (
	// launchCapacity bounds the launches waiting for their GPU record.
	launchCapacity = 1 << 17
	// stackCapacity bounds the distinct host stacks those launches refer to.
	stackCapacity = 1 << 14
	// processCapacity bounds the processes whose generation and metadata are
	// kept.
	processCapacity = 1 << 12
	// pendingCapacity bounds the distinct GPU work waiting for its launch.
	pendingCapacity = 1 << 16
	// unresolvedCapacity bounds the GPU work waiting for the UUID of its GPU.
	unresolvedCapacity = 1 << 16
	// gpuCapacity bounds the GPU indexes kept, of all processes.
	gpuCapacity = 1 << 14
	// samplesPerProcess bounds the samples of a process in each report.
	samplesPerProcess = 1 << 12
	// otherWork names the sample that work beyond samplesPerProcess goes to.
	otherWork = "[other GPU work]"
	// hwIDKey and hwTypeKey identify the GPU that did the work, as the
	// hw.gpu.* metrics do.
	hwIDKey   = attribute.Key("hw.id")
	hwTypeKey = attribute.Key("hw.type")
	// matchGrace is how long a GPU record waits for its launch stack, which
	// takes the slower path through the trace pipeline.
	matchGrace = 2 * time.Second
	// exitGrace is how long the state of an exited process is kept, for its
	// records and launch stacks still in flight.
	exitGrace = 5 * time.Second
	// gpuGrace is how long GPU work waits for the UUID of its GPU's index, which
	// producers send again at least every 10 seconds.
	gpuGrace = 12 * time.Second
)

// gpuWork is GPU activity to attribute to the host stack of its launch.
type gpuWork struct {
	pid libpf.PID
	// correlationID is the id of the call, 0 for work that no call started.
	correlationID uint64
	// sampleWeight is the number of calls the work's call stands for.
	sampleWeight uint32
	// frameType is the type of the frames added for the work, from the
	// runtime that reported it.
	frameType libpf.FrameType
	// leaf names the work in the leaf frame, e.g. the kernel name.
	leaf string
	// ranges is the stack of annotation ranges active at the call, such as
	// NVTX ranges, or empty.
	ranges string
	// gpu is the UUID of the GPU that did the work, or empty.
	gpu string
	// gpuIndex is the index the process names the GPU by, until the matcher
	// resolves it to gpu.
	gpuIndex uint32
	values   []profileValue
}

// profileValue is what gpuWork adds to a profile.
type profileValue struct {
	profileType *samples.TypeMetadata
	value       int64
}

// processKey identifies one process: each process gets its own generation, so
// a reused PID is never mistaken for the previous process.
type processKey struct {
	pid        libpf.PID
	generation uint32
}

func (k processKey) hash32() uint32 {
	return uint32(k.pid)*0x9e3779b1 ^ k.generation*0xc2b2ae35
}

// launchKey identifies the GPU call that issued some GPU work.
type launchKey struct {
	processKey
	correlationID uint64
}

func (k launchKey) hash32() uint32 {
	return k.processKey.hash32() ^ uint32(k.correlationID)*0x85ebca6b ^
		uint32(k.correlationID>>32)*0x27d4eb2f
}

// gpuKey identifies the GPU that a process names by an index.
type gpuKey struct {
	processKey
	index uint32
}

func (k gpuKey) hash32() uint32 {
	return k.processKey.hash32() ^ k.index*0x85ebca6b
}

// bucketKey identifies one exported sample: GPU work of a process, below one
// host stack.
type bucketKey struct {
	profileType *samples.TypeMetadata
	pid         libpf.PID
	// stack is the hash of the host stack, zero if the launch was not captured.
	stack     libpf.TraceHash
	frameType libpf.FrameType
	leaf      string
	ranges    string
	gpu       string
}

type bucket struct {
	frames libpf.Frames
	hash   libpf.TraceHash
	attrs  []attribute.KeyValue
	value  int64
	// meta is the process metadata captured while the process ran, or nil.
	meta *samples.ProcessMeta
}

// workKey identifies the GPU work of one launch that is merged while it waits
// for the launch stack: a CUDA graph launch runs hundreds of kernels, mostly
// repeating a few.
type workKey struct {
	frameType libpf.FrameType
	leaf      string
	ranges    string
	gpu       string
}

// pendingWork is GPU work whose launch stack has not arrived yet.
type pendingWork struct {
	work gpuWork
	// arrived is when the first of the merged work arrived.
	arrived time.Time
}

// matcher joins GPU records with the host stack captured at their launch,
// by process and correlation id, and aggregates them into snapshot samples.
type matcher struct {
	// processMeta looks up process metadata in the tracer. It must not be
	// called with mu held: Detach runs under the process manager's lock.
	processMeta func(libpf.PID) samples.ProcessMeta

	mu sync.Mutex
	// generations holds the generation of each process with GPU activity, and
	// lastGeneration the one given last. Work arriving after its process was
	// retired starts a generation that no exit retires: the table drops the
	// least recently active processes when full.
	generations    *lru.LRU[libpf.PID, uint32]
	lastGeneration uint32
	// exits holds when processes with GPU activity exited, until their state
	// is retired after exitGrace.
	exits map[libpf.PID]time.Time
	// launches holds the host stack of each launch.
	launches *lru.LRU[launchKey, libpf.TraceHash]
	stacks   *lru.LRU[libpf.TraceHash, libpf.Frames]
	// metas holds the metadata of processes with GPU activity, captured while
	// they run, and retiredMetas that of retired processes by PID, both for
	// samples reported after the process exited.
	metas        *lru.LRU[processKey, *samples.ProcessMeta]
	retiredMetas *lru.LRU[libpf.PID, *samples.ProcessMeta]
	// gpus holds the UUIDs of the GPUs that processes name by an index, and
	// unresolved the work naming an index whose UUID has not arrived yet.
	gpus          *lru.LRU[gpuKey, string]
	unresolved    map[gpuKey][]pendingWork
	numUnresolved int
	pending       map[launchKey]map[workKey]*pendingWork
	numPending    int
	buckets       map[bucketKey]*bucket
	// samples counts the buckets of each process since the previous report.
	samples map[libpf.PID]int
}

func newMatcher(processMeta func(libpf.PID) samples.ProcessMeta) (*matcher, error) {
	generations, err := lru.New[libpf.PID, uint32](processCapacity, libpf.PID.Hash32)
	if err != nil {
		return nil, err
	}
	launches, err := lru.New[launchKey, libpf.TraceHash](launchCapacity, launchKey.hash32)
	if err != nil {
		return nil, err
	}
	stacks, err := lru.New[libpf.TraceHash, libpf.Frames](stackCapacity, libpf.TraceHash.Hash32)
	if err != nil {
		return nil, err
	}
	metas, err := lru.New[processKey, *samples.ProcessMeta](processCapacity, processKey.hash32)
	if err != nil {
		return nil, err
	}
	retiredMetas, err := lru.New[libpf.PID, *samples.ProcessMeta](processCapacity, libpf.PID.Hash32)
	if err != nil {
		return nil, err
	}
	gpus, err := lru.New[gpuKey, string](gpuCapacity, gpuKey.hash32)
	if err != nil {
		return nil, err
	}
	return &matcher{
		processMeta:  processMeta,
		generations:  generations,
		exits:        make(map[libpf.PID]time.Time),
		launches:     launches,
		stacks:       stacks,
		metas:        metas,
		retiredMetas: retiredMetas,
		gpus:         gpus,
		unresolved:   make(map[gpuKey][]pendingWork),
		pending:      make(map[launchKey]map[workKey]*pendingWork),
		buckets:      make(map[bucketKey]*bucket),
		samples:      make(map[libpf.PID]int),
	}, nil
}

// processKeyLocked returns the key of the current process with this PID, which
// it marks as having GPU activity. The caller must hold m.mu.
func (m *matcher) processKeyLocked(pid libpf.PID, now time.Time) processKey {
	if exited, ok := m.exits[pid]; ok && now.Sub(exited) >= exitGrace {
		m.retireLocked(pid)
	}
	generation, ok := m.generations.Get(pid)
	if !ok {
		m.lastGeneration++
		generation = m.lastGeneration
		m.generations.Add(pid, generation)
	}
	return processKey{pid: pid, generation: generation}
}

func (m *matcher) launchKeyLocked(pid libpf.PID, correlationID uint64, now time.Time) launchKey {
	return launchKey{processKey: m.processKeyLocked(pid, now), correlationID: correlationID}
}

// addLaunch records the host stack of a GPU call and folds the GPU records
// that arrived before it.
func (m *matcher) addLaunch(pid libpf.PID, correlationID uint64, trace *libpf.Trace,
	now time.Time,
) {
	m.rememberMeta(pid, now)
	stack := trace.Hash()

	m.mu.Lock()
	defer m.mu.Unlock()
	key := m.launchKeyLocked(pid, correlationID, now)
	m.stacks.Add(stack, trace.Frames)

	// The launch is kept even once work arrived: more may follow, like the
	// other kernels of a graph. The table drops the oldest launches when full.
	m.launches.Add(key, stack)
	pending, ok := m.pending[key]
	if !ok {
		return
	}
	delete(m.pending, key)
	m.numPending -= len(pending)
	for _, p := range pending {
		m.foldLocked(p.work, stack, trace.Frames)
	}
}

// addWork folds GPU work, multiplied by its sample weight, into its launch
// stack's samples, or keeps it until the launch and the UUID of its GPU arrive.
func (m *matcher) addWork(work gpuWork, now time.Time) {
	if len(work.values) == 0 {
		return
	}
	m.rememberMeta(work.pid, now)

	m.mu.Lock()
	defer m.mu.Unlock()
	m.addWorkLocked(work, m.processKeyLocked(work.pid, now), now)
}

// addWorkLocked adds the work of process pk. The caller must hold m.mu.
func (m *matcher) addWorkLocked(work gpuWork, pk processKey, now time.Time) {
	if work.gpuIndex != 0 {
		gk := gpuKey{processKey: pk, index: work.gpuIndex}
		if uuid, ok := m.gpus.Get(gk); ok {
			work.gpu = uuid
		} else if m.numUnresolved < unresolvedCapacity {
			m.unresolved[gk] = append(m.unresolved[gk], pendingWork{work: work, arrived: now})
			m.numUnresolved++
			return
		}
		work.gpuIndex = 0
	}
	work = work.scaled()
	if work.correlationID == 0 {
		// No call started the work, so no stack will come.
		m.foldLocked(work, libpf.TraceHash{}, nil)
		return
	}
	key := launchKey{processKey: pk, correlationID: work.correlationID}
	if stack, ok := m.launches.Get(key); ok {
		if frames, ok := m.stacks.Get(stack); ok {
			m.foldLocked(work, stack, frames)
		} else {
			// The stack was evicted and will not arrive again.
			m.foldLocked(work, libpf.TraceHash{}, nil)
		}
		return
	}
	works := m.pending[key]
	wk := workKey{frameType: work.frameType, leaf: work.leaf, ranges: work.ranges, gpu: work.gpu}
	if p, ok := works[wk]; ok {
		p.work.merge(work)
		return
	}
	if m.numPending >= pendingCapacity {
		m.foldLocked(work, libpf.TraceHash{}, nil)
		return
	}
	if works == nil {
		works = make(map[workKey]*pendingWork)
		m.pending[key] = works
	}
	works[wk] = &pendingWork{work: work, arrived: now}
	m.numPending++
}

// foldLocked adds the work's values to the samples of its leaf below the host
// stack. The caller must hold m.mu.
func (m *matcher) foldLocked(work gpuWork, stack libpf.TraceHash, hostFrames libpf.Frames) {
	if m.samples[work.pid] >= samplesPerProcess {
		work = otherWorkOf(work)
		stack, hostFrames = libpf.TraceHash{}, nil
	}
	for _, v := range work.values {
		key := bucketKey{
			profileType: v.profileType,
			pid:         work.pid,
			stack:       stack,
			frameType:   work.frameType,
			leaf:        work.leaf,
			ranges:      work.ranges,
			gpu:         work.gpu,
		}
		b, ok := m.buckets[key]
		if !ok {
			b = &bucket{
				frames: sampleFrames(work, hostFrames),
				hash:   sampleHash(stack, work),
				attrs:  sampleAttributes(work),
				meta:   m.capturedMetaLocked(work.pid),
			}
			m.buckets[key] = b
			m.samples[work.pid]++
		}
		b.value = saturatingAdd(b.value, v.value)
	}
}

// otherWorkOf returns the work as part of the single sample that a process's
// work beyond samplesPerProcess goes to.
func otherWorkOf(work gpuWork) gpuWork {
	work.leaf = otherWork
	work.ranges = ""
	work.gpu = ""
	return work
}

// rangeSeparator separates the ranges of a range stack.
const rangeSeparator = "\x1f"

// sampleFrames returns the frames of a sample: the work, the annotation ranges
// it ran in from the innermost outwards, then the host stack.
func sampleFrames(work gpuWork, hostFrames libpf.Frames) libpf.Frames {
	frames := make(libpf.Frames, 0, len(hostFrames)+2)
	frames.Append(&libpf.Frame{Type: work.frameType, FunctionName: libpf.Intern(work.leaf)})
	for name := range strings.SplitSeq(work.ranges, rangeSeparator) {
		if name != "" {
			frames.Append(&libpf.Frame{
				Type:         work.frameType,
				FunctionName: libpf.Intern("range:" + name),
			})
		}
	}
	return append(frames, hostFrames...)
}

// capturedMetaLocked returns the metadata captured for the current process
// with this PID or, once it was retired, for the last one. The caller must hold
// m.mu.
func (m *matcher) capturedMetaLocked(pid libpf.PID) *samples.ProcessMeta {
	if generation, ok := m.generations.Peek(pid); ok {
		if meta, ok := m.metas.Get(processKey{pid: pid, generation: generation}); ok {
			return meta
		}
	}
	meta, _ := m.retiredMetas.Get(pid)
	return meta
}

// sampleAttributes returns the attributes of a sample, or nil.
func sampleAttributes(work gpuWork) []attribute.KeyValue {
	if work.gpu == "" {
		return nil
	}
	return []attribute.KeyValue{hwIDKey.String(work.gpu), hwTypeKey.String("gpu")}
}

// sampleHash identifies a sample. libpf.Trace.Hash does not cover function
// names, the only difference between synthesized frames, nor attributes.
func sampleHash(stack libpf.TraceHash, work gpuWork) libpf.TraceHash {
	h := xxh3.New128()
	_, _ = h.Write(stack.Bytes())
	_, _ = h.Write([]byte{byte(work.frameType)})
	// The separators keep "a"+"bc" apart from "ab"+"c".
	_, _ = h.WriteString(work.leaf)
	_, _ = h.WriteString(rangeSeparator)
	_, _ = h.WriteString(work.ranges)
	_, _ = h.WriteString(rangeSeparator)
	_, _ = h.WriteString(work.gpu)
	sum := h.Sum128()
	return libpf.NewTraceHash(sum.Hi, sum.Lo)
}

// expireUnresolvedLocked adds the GPU work whose GPU's UUID did not arrive
// within gpuGrace, without a GPU. The caller must hold m.mu.
func (m *matcher) expireUnresolvedLocked(now time.Time) {
	for gk, works := range m.unresolved {
		// Work is kept in the order it arrived.
		expired := 0
		for expired < len(works) && now.Sub(works[expired].arrived) >= gpuGrace {
			expired++
		}
		if expired == len(works) {
			delete(m.unresolved, gk)
		} else {
			m.unresolved[gk] = works[expired:]
		}
		m.numUnresolved -= expired
		for _, p := range works[:expired] {
			p.work.gpuIndex = 0
			m.addWorkLocked(p.work, gk.processKey, p.arrived)
		}
	}
}

// releaseUnresolvedLocked adds the work of a retired process that still waits
// for the UUID of its GPU, without one: the process will not send it. The caller
// must hold m.mu.
func (m *matcher) releaseUnresolvedLocked(pk processKey) {
	for gk, works := range m.unresolved {
		if gk.processKey != pk {
			continue
		}
		delete(m.unresolved, gk)
		m.numUnresolved -= len(works)
		for _, p := range works {
			p.work.gpuIndex = 0
			m.addWorkLocked(p.work, pk, p.arrived)
		}
	}
}

// expirePendingLocked folds GPU work whose launch did not arrive within
// matchGrace, attributed to the process only. The caller must hold m.mu.
func (m *matcher) expirePendingLocked(now time.Time) {
	for key, works := range m.pending {
		for wk, p := range works {
			if now.Sub(p.arrived) < matchGrace {
				continue
			}
			m.foldLocked(p.work, libpf.TraceHash{}, nil)
			delete(works, wk)
			m.numPending--
		}
		if len(works) == 0 {
			delete(m.pending, key)
		}
	}
}

// processExited schedules the state of an exited process to be retired once
// its records still in flight had time to arrive.
func (m *matcher) processExited(pid libpf.PID, now time.Time) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.generations.Contains(pid) {
		m.exits[pid] = now
	}
}

// processStarted retires the previous process with this PID at once if it
// exited: a new process reused the PID before exitGrace passed.
func (m *matcher) processStarted(pid libpf.PID) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, ok := m.exits[pid]; ok {
		m.retireLocked(pid)
	}
}

// retireLocked forgets the state of an exited process. Its buffered launches
// become unreachable, since a process reusing the PID gets a new generation.
// The caller must hold m.mu.
func (m *matcher) retireLocked(pid libpf.PID) {
	if generation, ok := m.generations.Peek(pid); ok {
		pk := processKey{pid: pid, generation: generation}
		if meta, ok := m.metas.Get(pk); ok {
			m.retiredMetas.Add(pid, meta)
		}
		m.releaseUnresolvedLocked(pk)
	}
	delete(m.exits, pid)
	m.generations.Remove(pid)
}

// retireExitedLocked retires the processes that exited more than exitGrace ago.
// The caller must hold m.mu.
func (m *matcher) retireExitedLocked(now time.Time) {
	for pid, exited := range m.exits {
		if now.Sub(exited) >= exitGrace {
			m.retireLocked(pid)
		}
	}
}

// rememberMeta captures the metadata of a process the first time it has GPU
// activity that the tracer knows the process, for samples reported after it
// exited.
func (m *matcher) rememberMeta(pid libpf.PID, now time.Time) {
	m.mu.Lock()
	key := m.processKeyLocked(pid, now)
	known := m.metas.Contains(key)
	m.mu.Unlock()
	if known {
		return
	}

	meta := m.processMeta(pid)
	if meta.ExecutablePath == libpf.NullString {
		// The tracer has not loaded the process yet: try again later.
		return
	}
	m.mu.Lock()
	m.metas.Add(key, &meta)
	m.mu.Unlock()
}

// setGPU records the UUID of the GPU that a process names by index, and adds
// the work that waited for it.
func (m *matcher) setGPU(pid libpf.PID, index uint32, uuid string, now time.Time) {
	m.mu.Lock()
	defer m.mu.Unlock()
	gk := gpuKey{processKey: m.processKeyLocked(pid, now), index: index}
	m.gpus.Add(gk, uuid)
	works := m.unresolved[gk]
	delete(m.unresolved, gk)
	m.numUnresolved -= len(works)
	for _, p := range works {
		m.addWorkLocked(p.work, gk.processKey, p.arrived)
	}
}

// snapshot returns the samples aggregated since the previous call.
func (m *matcher) snapshot(now time.Time) []samples.SnapshotProfile {
	m.mu.Lock()
	m.expireUnresolvedLocked(now)
	m.expirePendingLocked(now)
	m.retireExitedLocked(now)
	buckets := m.buckets
	m.buckets = make(map[bucketKey]*bucket)
	clear(m.samples)
	m.mu.Unlock()

	fresh := make(map[libpf.PID]samples.ProcessMeta)
	profiles := make(map[*samples.TypeMetadata]*samples.SnapshotProfile)
	for key, b := range buckets {
		meta, ok := fresh[key.pid]
		if !ok {
			meta = m.processMeta(key.pid)
			fresh[key.pid] = meta
		}
		sampleMeta := &meta
		if meta.ExecutablePath == libpf.NullString && b.meta != nil {
			// The process exited and the tracer forgot it.
			sampleMeta = b.meta
		}

		profile, ok := profiles[key.profileType]
		if !ok {
			profile = &samples.SnapshotProfile{ProfileType: key.profileType}
			profiles[key.profileType] = profile
		}
		profile.Samples = append(profile.Samples, samples.SnapshotSample{
			PID:         key.pid,
			TraceHash:   b.hash,
			Frames:      b.frames,
			Attributes:  b.attrs,
			Value:       b.value,
			ProcessMeta: sampleMeta,
		})
	}

	result := make([]samples.SnapshotProfile, 0, len(profiles))
	for _, profile := range profiles {
		result = append(result, *profile)
	}
	return result
}
