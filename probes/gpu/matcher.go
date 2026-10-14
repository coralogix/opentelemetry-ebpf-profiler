// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package gpu // import "go.opentelemetry.io/ebpf-profiler/probes/gpu"

import (
	"strings"
	"sync"
	"time"

	lru "github.com/elastic/go-freelru"
	"github.com/zeebo/xxh3"

	"go.opentelemetry.io/ebpf-profiler/libpf"
	"go.opentelemetry.io/ebpf-profiler/reporter/samples"
)

const (
	// launchCapacity bounds the launches waiting for their GPU record.
	launchCapacity = 1 << 17
	// stackCapacity bounds the distinct host stacks those launches refer to.
	stackCapacity = 1 << 14
	// processCapacity bounds the processes whose metadata is kept.
	processCapacity = 1 << 12
	// pendingCapacity bounds the distinct GPU work waiting for its launch.
	pendingCapacity = 1 << 16
	// matchGrace is how long a GPU record waits for its launch stack, which
	// takes the slower path through the trace pipeline.
	matchGrace = 2 * time.Second
	// exitGrace is how long the state of an exited process is kept, for its
	// records and launch stacks still in flight.
	exitGrace = 5 * time.Second
)

// gpuWork is GPU activity to attribute to the host stack of its launch.
type gpuWork struct {
	pid           libpf.PID
	correlationID uint32
	// leaf names the work in the leaf frame, e.g. the kernel name.
	leaf string
	// ranges is the stack of annotation ranges active at the call site, such
	// as NVTX ranges, or empty.
	ranges string
	values []profileValue
	// consumesLaunch is set for the only work of its launch, a kernel
	// launched outside of a CUDA graph.
	consumesLaunch bool
}

// profileValue is what gpuWork adds to a profile.
type profileValue struct {
	profileType *samples.TypeMetadata
	value       int64
}

// processKey identifies one process: generation changes when the PID exits,
// so a reused PID is never mistaken for the previous process.
type processKey struct {
	pid        libpf.PID
	generation uint32
}

func (k processKey) hash32() uint32 {
	return uint32(k.pid)*0x9e3779b1 ^ k.generation*0xc2b2ae35
}

// launchKey identifies the GPU call site that issued some GPU work.
type launchKey struct {
	processKey
	correlationID uint32
}

func (k launchKey) hash32() uint32 {
	return k.processKey.hash32() ^ k.correlationID*0x85ebca6b
}

// bucketKey identifies one exported sample: GPU work of a process, below one
// host stack.
type bucketKey struct {
	profileType *samples.TypeMetadata
	pid         libpf.PID
	// stack is the hash of the host stack, zero if the launch was not captured.
	stack  libpf.TraceHash
	leaf   string
	ranges string
}

type bucket struct {
	frames libpf.Frames
	hash   libpf.TraceHash
	value  int64
	// meta is the process metadata captured while the process ran, or nil.
	meta *samples.ProcessMeta
}

// workKey identifies the GPU work of one launch that is merged while it waits
// for the launch stack: a CUDA graph launch runs hundreds of kernels, mostly
// repeating a few.
type workKey struct {
	leaf   string
	ranges string
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
	names       *nameCache

	mu sync.Mutex
	// generations holds the current generation of each PID with GPU activity.
	generations map[libpf.PID]uint32
	// exits holds when processes with GPU activity exited, until their state
	// is retired after exitGrace.
	exits    map[libpf.PID]time.Time
	launches *lru.LRU[launchKey, libpf.TraceHash]
	stacks   *lru.LRU[libpf.TraceHash, libpf.Frames]
	// metas holds the metadata of processes seen launching GPU work, for
	// samples reported after the process exited.
	metas      *lru.LRU[processKey, *samples.ProcessMeta]
	pending    map[launchKey]map[workKey]*pendingWork
	numPending int
	buckets    map[bucketKey]*bucket
}

func newMatcher(processMeta func(libpf.PID) samples.ProcessMeta, names *nameCache,
) (*matcher, error) {
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
	return &matcher{
		processMeta: processMeta,
		names:       names,
		generations: make(map[libpf.PID]uint32),
		exits:       make(map[libpf.PID]time.Time),
		launches:    launches,
		stacks:      stacks,
		metas:       metas,
		pending:     make(map[launchKey]map[workKey]*pendingWork),
		buckets:     make(map[bucketKey]*bucket),
	}, nil
}

// processKeyLocked returns the key of the current process with this PID, which
// it marks as having GPU activity. The caller must hold m.mu.
func (m *matcher) processKeyLocked(pid libpf.PID, now time.Time) processKey {
	if exited, ok := m.exits[pid]; ok && now.Sub(exited) >= exitGrace {
		m.retireLocked(pid)
	}
	generation, ok := m.generations[pid]
	if !ok {
		m.generations[pid] = 0
	}
	return processKey{pid: pid, generation: generation}
}

func (m *matcher) launchKeyLocked(pid libpf.PID, correlationID uint32, now time.Time) launchKey {
	return launchKey{processKey: m.processKeyLocked(pid, now), correlationID: correlationID}
}

// addLaunch records the host stack of a GPU call site and folds the GPU
// records that arrived before it.
func (m *matcher) addLaunch(pid libpf.PID, correlationID uint32, trace *libpf.Trace,
	now time.Time,
) {
	m.rememberMeta(pid, now)
	stack := trace.Hash()

	m.mu.Lock()
	defer m.mu.Unlock()
	key := m.launchKeyLocked(pid, correlationID, now)
	m.stacks.Add(stack, trace.Frames)

	pending, ok := m.pending[key]
	if !ok {
		m.launches.Add(key, stack)
		return
	}
	delete(m.pending, key)
	m.numPending -= len(pending)
	consumed := false
	for _, p := range pending {
		m.foldLocked(p.work, stack, trace.Frames)
		consumed = consumed || p.work.consumesLaunch
	}
	if !consumed {
		// More work of the launch may follow, like other kernels of a graph.
		m.launches.Add(key, stack)
	}
}

// addWork folds GPU work into its launch stack's samples, or keeps it until the
// launch arrives.
func (m *matcher) addWork(work gpuWork, now time.Time) {
	if len(work.values) == 0 {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	key := m.launchKeyLocked(work.pid, work.correlationID, now)
	if stack, ok := m.launches.Get(key); ok {
		if frames, ok := m.stacks.Get(stack); ok {
			if work.consumesLaunch {
				m.launches.Remove(key)
			}
			m.foldLocked(work, stack, frames)
			return
		}
	}
	works := m.pending[key]
	wk := workKey{leaf: work.leaf, ranges: work.ranges}
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
	for _, v := range work.values {
		key := bucketKey{
			profileType: v.profileType,
			pid:         work.pid,
			stack:       stack,
			leaf:        work.leaf,
			ranges:      work.ranges,
		}
		b, ok := m.buckets[key]
		if !ok {
			b = &bucket{
				frames: sampleFrames(work, hostFrames),
				hash:   sampleHash(stack, work),
				meta:   m.capturedMetaLocked(work.pid),
			}
			m.buckets[key] = b
		}
		b.value += v.value
	}
}

// rangeSeparator separates the ranges of a range stack.
const rangeSeparator = "\x1f"

// sampleFrames returns the frames of a sample: the work, the annotation ranges
// it ran in from the innermost outwards, then the host stack.
func sampleFrames(work gpuWork, hostFrames libpf.Frames) libpf.Frames {
	frames := make(libpf.Frames, 0, len(hostFrames)+1)
	frames.Append(&libpf.Frame{Type: libpf.GPUFrame, FunctionName: libpf.Intern(work.leaf)})
	for name := range strings.SplitSeq(work.ranges, rangeSeparator) {
		if name != "" {
			frames.Append(&libpf.Frame{
				Type:         libpf.GPUFrame,
				FunctionName: libpf.Intern("range:" + name),
			})
		}
	}
	return append(frames, hostFrames...)
}

// capturedMetaLocked returns the metadata captured for the current or, after
// it exited, the previous process with this PID. The caller must hold m.mu.
func (m *matcher) capturedMetaLocked(pid libpf.PID) *samples.ProcessMeta {
	generation := m.generations[pid]
	if meta, ok := m.metas.Get(processKey{pid: pid, generation: generation}); ok {
		return meta
	}
	if generation == 0 {
		return nil
	}
	meta, _ := m.metas.Get(processKey{pid: pid, generation: generation - 1})
	return meta
}

// sampleHash identifies a sample's stack. libpf.Trace.Hash does not cover
// function names, the only difference between synthesized frames.
func sampleHash(stack libpf.TraceHash, work gpuWork) libpf.TraceHash {
	h := xxh3.New128()
	_, _ = h.Write(stack.Bytes())
	// The separators keep "a"+"bc" apart from "ab"+"c".
	_, _ = h.WriteString(work.leaf)
	_, _ = h.WriteString(rangeSeparator)
	_, _ = h.WriteString(work.ranges)
	sum := h.Sum128()
	return libpf.NewTraceHash(sum.Hi, sum.Lo)
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
	if _, ok := m.generations[pid]; ok {
		m.exits[pid] = now
	}
}

// retireLocked forgets the state of an exited process. Its buffered launches
// become unreachable under the next generation, so a process reusing the PID
// never matches them. The caller must hold m.mu.
func (m *matcher) retireLocked(pid libpf.PID) {
	delete(m.exits, pid)
	m.generations[pid]++
	m.names.forget(pid)
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

// rememberMeta captures the metadata of a process the first time it launches
// GPU work, for samples reported after it exited.
func (m *matcher) rememberMeta(pid libpf.PID, now time.Time) {
	m.mu.Lock()
	key := m.processKeyLocked(pid, now)
	known := m.metas.Contains(key)
	m.mu.Unlock()
	if known {
		return
	}

	meta := m.processMeta(pid)
	m.mu.Lock()
	m.metas.Add(key, &meta)
	m.mu.Unlock()
}

// snapshot returns the samples aggregated since the previous call.
func (m *matcher) snapshot(now time.Time) []samples.SnapshotProfile {
	m.mu.Lock()
	m.expirePendingLocked(now)
	m.retireExitedLocked(now)
	buckets := m.buckets
	m.buckets = make(map[bucketKey]*bucket)
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
