// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package gpu

import (
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"go.opentelemetry.io/ebpf-profiler/libpf"
	"go.opentelemetry.io/ebpf-profiler/reporter/samples"
)

const testPID = libpf.PID(100)

func hostTrace(addresses ...libpf.AddressOrLineno) *libpf.Trace {
	frames := make(libpf.Frames, 0, len(addresses))
	for _, address := range addresses {
		frames.Append(&libpf.Frame{Type: libpf.NativeFrame, AddressOrLineno: address})
	}
	return &libpf.Trace{Frames: frames}
}

func kernel(correlationID uint64, duration uint64) event {
	return event{pid: testPID, kind: kindKernel, runtime: runtimeCUDA,
		correlationID: correlationID, durationNS: duration}
}

// weighted returns ev with the sample weight of its call.
func weighted(ev event, sampleWeight uint32) event {
	ev.sampleWeight = sampleWeight
	return ev
}

// addEvent adds the work of an event to m.
func addEvent(t *testing.T, m *matcher, ev event, now time.Time) {
	t.Helper()
	work, ok := eventWork(ev)
	require.True(t, ok)
	m.addWork(work, now)
}

// addKernel adds a kernel execution to m.
func addKernel(t *testing.T, m *matcher, ev event, name string, now time.Time) {
	t.Helper()
	ev.name = name
	addEvent(t, m, ev, now)
}

// liveProcesses fakes the tracer's process metadata for the PIDs in it.
type liveProcesses map[libpf.PID]string

func (l liveProcesses) meta(pid libpf.PID) samples.ProcessMeta {
	if exe, ok := l[pid]; ok {
		return samples.ProcessMeta{ExecutablePath: libpf.Intern(exe)}
	}
	return samples.ProcessMeta{}
}

func newTestMatcher(t *testing.T, live liveProcesses) *matcher {
	t.Helper()
	m, err := newMatcher(live.meta)
	require.NoError(t, err)
	return m
}

// snapshotSamples returns the kernel time samples by leaf name.
func snapshotSamples(t *testing.T, m *matcher, now time.Time) map[string]samples.SnapshotSample {
	t.Helper()
	bySample := make(map[string]samples.SnapshotSample)
	for _, profile := range m.snapshot(now) {
		require.Same(t, kernelTimeProfileType, profile.ProfileType)
		for _, s := range profile.Samples {
			leaf := s.Frames[0].Value()
			require.Equal(t, libpf.CUDAFrame, leaf.Type)
			key := leaf.FunctionName.String()
			if len(s.Frames) == 1 {
				key += " (no stack)"
			}
			bySample[key] = s
		}
	}
	return bySample
}

func TestMatcherJoinsKernelsWithLaunches(t *testing.T) {
	m := newTestMatcher(t, liveProcesses{testPID: "/bin/app"})
	now := time.Now()

	m.addLaunch(testPID, 1, hostTrace(0x10, 0x20), now)
	addKernel(t, m, kernel(1, 100), "vector_add", now)
	// Arrives before its launch.
	addKernel(t, m, kernel(2, 50), "vector_add", now)
	m.addLaunch(testPID, 2, hostTrace(0x10, 0x20), now)
	m.addLaunch(testPID, 3, hostTrace(0x30), now)
	addKernel(t, m, kernel(3, 7), "reduce", now)

	got := snapshotSamples(t, m, now)
	require.Len(t, got, 2)
	assert.Equal(t, int64(150), got["vector_add"].Value)
	assert.Len(t, got["vector_add"].Frames, 3)
	assert.Equal(t, int64(7), got["reduce"].Value)
	assert.NotEqual(t, got["vector_add"].TraceHash, got["reduce"].TraceHash)
	assert.Equal(t, "/bin/app", got["reduce"].ProcessMeta.ExecutablePath.String())

	assert.Empty(t, m.snapshot(now), "samples are reported once")
}

func TestMatcherSampleWeight(t *testing.T) {
	m := newTestMatcher(t, liveProcesses{testPID: "/bin/app"})
	now := time.Now()
	m.addLaunch(testPID, 1, hostTrace(0x10), now)
	addKernel(t, m, weighted(kernel(1, 10), 3), "joined", now)
	// Arrives before its launch.
	addKernel(t, m, weighted(kernel(2, 10), 4), "pending", now)
	m.addLaunch(testPID, 2, hostTrace(0x10), now)
	// Its launch never arrives.
	addKernel(t, m, weighted(kernel(3, 10), 2), "expired", now)

	got := snapshotSamples(t, m, now.Add(matchGrace))
	require.Len(t, got, 3)
	assert.Equal(t, int64(30), got["joined"].Value)
	assert.Equal(t, int64(40), got["pending"].Value)
	assert.Equal(t, int64(20), got["expired (no stack)"].Value, "keeps its weight without a stack")
}

func TestMatcherEvictedStackKeepsWeight(t *testing.T) {
	m := newTestMatcher(t, liveProcesses{testPID: "/bin/app"})
	now := time.Now()
	trace := hostTrace(0x10)
	m.addLaunch(testPID, 1, trace, now)
	m.stacks.Remove(trace.Hash())
	addKernel(t, m, weighted(kernel(1, 10), 3), "k", now)

	got := snapshotSamples(t, m, now)
	require.Len(t, got, 1, "reported without waiting for the launch")
	assert.Equal(t, int64(30), got["k (no stack)"].Value)
}

func TestMatcherSplitsByGPU(t *testing.T) {
	m := newTestMatcher(t, liveProcesses{testPID: "/bin/app"})
	now := time.Now()
	for i, gpu := range []string{"GPU-a", "GPU-b", ""} {
		m.addLaunch(testPID, uint64(i+1), hostTrace(0x10), now)
		ev := kernel(uint64(i+1), 10)
		ev.gpu = gpu
		addKernel(t, m, ev, "k", now)
	}

	profiles := m.snapshot(now)
	require.Len(t, profiles, 1)
	require.Len(t, profiles[0].Samples, 3, "GPUs split samples")
	byGPU := make(map[string]libpf.TraceHash)
	for _, s := range profiles[0].Samples {
		gpu := ""
		for _, attr := range s.Attributes {
			if attr.Key == hwIDKey {
				gpu = attr.Value.AsString()
			}
		}
		byGPU[gpu] = s.TraceHash
	}
	assert.Len(t, byGPU, 3)
	assert.NotEqual(t, byGPU["GPU-a"], byGPU["GPU-b"])
	assert.NotEqual(t, byGPU["GPU-a"], byGPU[""])
}

func TestMatcherSameStackDifferentKernels(t *testing.T) {
	m := newTestMatcher(t, liveProcesses{testPID: "/bin/app"})
	now := time.Now()
	m.addLaunch(testPID, 1, hostTrace(0x10), now)
	addKernel(t, m, kernel(1, 1), "a", now)
	m.addLaunch(testPID, 2, hostTrace(0x10), now)
	addKernel(t, m, kernel(2, 1), "b", now)

	got := snapshotSamples(t, m, now)
	require.Len(t, got, 2)
	assert.NotEqual(t, got["a"].TraceHash, got["b"].TraceHash)
}

func TestMatcherGraphLaunch(t *testing.T) {
	m := newTestMatcher(t, liveProcesses{testPID: "/bin/app"})
	now := time.Now()
	// The kernels of a graph share the call id of its launch.
	graphKernel := kernel(5, 10)

	addKernel(t, m, graphKernel, "step", now)
	m.addLaunch(testPID, 5, hostTrace(0x10), now)
	addKernel(t, m, graphKernel, "step", now)
	addKernel(t, m, graphKernel, "step", now)

	got := snapshotSamples(t, m, now)
	require.Len(t, got, 1)
	assert.Equal(t, int64(30), got["step"].Value)
}

func TestMatcherUnmatchedKernelsExpire(t *testing.T) {
	m := newTestMatcher(t, liveProcesses{testPID: "/bin/app"})
	now := time.Now()
	addKernel(t, m, kernel(1, 40), "orphan", now)

	assert.Empty(t, m.snapshot(now), "kernel waits for its launch")
	got := snapshotSamples(t, m, now.Add(matchGrace))
	require.Len(t, got, 1)
	assert.Equal(t, int64(40), got["orphan (no stack)"].Value)
}

func TestMatcherInvalidKernel(t *testing.T) {
	m := newTestMatcher(t, liveProcesses{testPID: "/bin/app"})
	now := time.Now()
	m.addLaunch(testPID, 1, hostTrace(0x10), now)
	addKernel(t, m, kernel(1, 0), "k", now)
	assert.Empty(t, m.snapshot(now.Add(matchGrace)))
}

func TestMatcherReusedPID(t *testing.T) {
	m := newTestMatcher(t, liveProcesses{testPID: "/bin/app"})
	now := time.Now()
	m.addLaunch(testPID, 1, hostTrace(0x10), now)
	m.processExited(testPID, now)

	// A new process reuses the PID and the correlation id after exitGrace.
	later := now.Add(exitGrace)
	addKernel(t, m, kernel(1, 10), "k", later)
	got := snapshotSamples(t, m, later.Add(matchGrace))
	require.Len(t, got, 1)
	_, ok := got["k (no stack)"]
	assert.True(t, ok, "must not match the previous process's launch")
}

func TestMatcherReusedPIDWithinExitGrace(t *testing.T) {
	m := newTestMatcher(t, liveProcesses{testPID: "/bin/app"})
	now := time.Now()
	m.addLaunch(testPID, 1, hostTrace(0x10), now)
	m.processExited(testPID, now)

	// A new process reuses the PID, and the correlation id, at once.
	m.processStarted(testPID)
	addKernel(t, m, kernel(1, 10), "k", now)
	got := snapshotSamples(t, m, now.Add(matchGrace))
	require.Len(t, got, 1)
	_, ok := got["k (no stack)"]
	assert.True(t, ok, "must not match the previous process's launch")
}

func TestMatcherForgetsRetiredProcesses(t *testing.T) {
	m := newTestMatcher(t, liveProcesses{testPID: "/bin/app"})
	now := time.Now()
	m.addLaunch(testPID, 1, hostTrace(0x10), now)
	m.processExited(testPID, now)
	m.snapshot(now.Add(exitGrace))
	assert.Zero(t, m.generations.Len())
	assert.Empty(t, m.exits)
}

func TestMatcherMatchesInFlightRecordsOfExitedProcess(t *testing.T) {
	m := newTestMatcher(t, liveProcesses{testPID: "/bin/app"})
	now := time.Now()
	addKernel(t, m, kernel(1, 10), "k", now)
	m.processExited(testPID, now)
	// The launch stack arrives through the trace pipeline after the exit.
	m.addLaunch(testPID, 1, hostTrace(0x10), now.Add(time.Second))

	got := snapshotSamples(t, m, now.Add(time.Second))
	require.Len(t, got, 1)
	assert.Len(t, got["k"].Frames, 2)
}

func TestMatcherKeepsMetadataOfExitedProcess(t *testing.T) {
	live := liveProcesses{testPID: "/bin/short-job"}
	m := newTestMatcher(t, live)
	now := time.Now()
	m.addLaunch(testPID, 1, hostTrace(0x10), now)
	addKernel(t, m, kernel(1, 10), "k", now)

	// The process exits and the tracer forgets it before the report.
	m.processExited(testPID, now)
	delete(live, testPID)

	got := snapshotSamples(t, m, now)
	require.Len(t, got, 1)
	assert.Equal(t, "/bin/short-job", got["k"].ProcessMeta.ExecutablePath.String())
}

func TestMatcherKeepsMetadataOfProcessWithoutStacks(t *testing.T) {
	live := liveProcesses{testPID: "/bin/short-job"}
	m := newTestMatcher(t, live)
	now := time.Now()
	// The launch never arrives: the process exited before its stack was taken.
	addKernel(t, m, kernel(1, 10), "k", now)

	m.processExited(testPID, now)
	delete(live, testPID)

	got := snapshotSamples(t, m, now.Add(matchGrace))
	require.Len(t, got, 1)
	assert.Equal(t, "/bin/short-job", got["k (no stack)"].ProcessMeta.ExecutablePath.String())
}

func TestMatcherUnknownKernelName(t *testing.T) {
	m := newTestMatcher(t, liveProcesses{testPID: "/bin/app"})
	now := time.Now()
	m.addLaunch(testPID, 1, hostTrace(0x10), now)
	addKernel(t, m, kernel(1, 1), "", now)
	_, ok := snapshotSamples(t, m, now)[unknownKernel]
	assert.True(t, ok)
}

func TestMatcherWorkSharesLaunch(t *testing.T) {
	m := newTestMatcher(t, liveProcesses{testPID: "/bin/app"})
	now := time.Now()
	m.addLaunch(testPID, 1, hostTrace(0x10), now)
	// A graph's copy and kernel share the call id.
	addEvent(t, m, event{pid: testPID, kind: kindCopyHtoD, correlationID: 1,
		durationNS: 5, bytes: 64}, now)
	addKernel(t, m, kernel(1, 3), "k", now)

	byType := make(map[string]int64)
	for _, profile := range m.snapshot(now) {
		for _, s := range profile.Samples {
			assert.Len(t, s.Frames, 2, "every value joins the launch stack")
			byType[profile.ProfileType.SampleTypes[0].Type] += s.Value
		}
	}
	assert.Equal(t, map[string]int64{
		"gpu_mem_time":    5,
		"gpu_mem_bytes":   64,
		"gpu_kernel_time": 3,
	}, byType)
}

func TestMatcherRanges(t *testing.T) {
	m := newTestMatcher(t, liveProcesses{testPID: "/bin/app"})
	now := time.Now()
	m.addLaunch(testPID, 1, hostTrace(0x10), now)
	ranged := kernel(1, 5)
	ranged.ranges = "aten::mm\x1faten::matmul\x1ftrain_step"
	addKernel(t, m, ranged, "k", now)
	m.addLaunch(testPID, 2, hostTrace(0x10), now)
	addKernel(t, m, kernel(2, 7), "k", now)

	profiles := m.snapshot(now)
	require.Len(t, profiles, 1)
	require.Len(t, profiles[0].Samples, 2, "ranges split samples")
	var names [][]string
	for _, s := range profiles[0].Samples {
		var frameNames []string
		for _, frame := range s.Frames {
			frameNames = append(frameNames, frame.Value().FunctionName.String())
		}
		names = append(names, frameNames)
	}
	assert.ElementsMatch(t, [][]string{
		{"k", "range:aten::mm", "range:aten::matmul", "range:train_step", ""},
		{"k", ""},
	}, names)
	assert.NotEqual(t, profiles[0].Samples[0].TraceHash, profiles[0].Samples[1].TraceHash)
}

func TestMatcherMergesPendingGraphWork(t *testing.T) {
	m := newTestMatcher(t, liveProcesses{testPID: "/bin/app"})
	now := time.Now()
	graphKernel := kernel(7, 2)
	// A graph launch runs more kernels than fit apart in the pending work.
	for range pendingCapacity + 10 {
		addKernel(t, m, graphKernel, "step", now)
	}
	addKernel(t, m, graphKernel, "reduce", now)
	assert.Equal(t, 2, m.numPending, "kernels of one launch and name wait as one")

	m.addLaunch(testPID, 7, hostTrace(0x10), now)
	got := snapshotSamples(t, m, now)
	require.Len(t, got, 2)
	assert.Equal(t, int64(2*(pendingCapacity+10)), got["step"].Value)
	assert.Equal(t, int64(2), got["reduce"].Value)
	assert.Zero(t, m.numPending)
}

func TestMatcherBoundsProcessesWithoutExit(t *testing.T) {
	m := newTestMatcher(t, liveProcesses{})
	now := time.Now()
	// Work arriving after its process was retired has no exit to retire it.
	for pid := range libpf.PID(2 * processCapacity) {
		ev := kernel(1, 100)
		ev.pid = pid + 1
		addKernel(t, m, ev, "late", now)
	}
	assert.Equal(t, processCapacity, m.generations.Len())
}

func TestMatcherWorkWithoutCall(t *testing.T) {
	m := newTestMatcher(t, liveProcesses{testPID: "/bin/app"})
	now := time.Now()
	// Work no call started is counted at once, without waiting for a stack.
	addKernel(t, m, kernel(0, 25), "device_launched", now)
	got := snapshotSamples(t, m, now)
	assert.Equal(t, int64(25), got["device_launched (no stack)"].Value)
	assert.Empty(t, m.pending)
}

func TestMatcherMatchesFullCorrelationID(t *testing.T) {
	m := newTestMatcher(t, liveProcesses{testPID: "/bin/app"})
	now := time.Now()
	m.addLaunch(testPID, 1<<32|7, hostTrace(0x10), now)
	// Same low half, another call.
	addKernel(t, m, kernel(2<<32|7, 10), "other", now)
	addKernel(t, m, kernel(1<<32|7, 20), "own", now)
	got := snapshotSamples(t, m, now.Add(matchGrace))
	assert.Equal(t, int64(20), got["own"].Value)
	assert.Equal(t, int64(10), got["other (no stack)"].Value)
}

func TestMatcherCapsSamplesPerProcess(t *testing.T) {
	m := newTestMatcher(t, liveProcesses{testPID: "/bin/app"})
	now := time.Now()
	m.addLaunch(testPID, 1, hostTrace(0x10), now)
	for i := range samplesPerProcess + 10 {
		addKernel(t, m, kernel(1, 1), fmt.Sprintf("k%d", i), now)
	}
	got := snapshotSamples(t, m, now)
	assert.Len(t, got, samplesPerProcess+1)
	assert.Equal(t, int64(10), got[otherWork+" (no stack)"].Value)

	// The cap starts over with each report.
	addKernel(t, m, kernel(1, 1), "next", now)
	assert.Contains(t, snapshotSamples(t, m, now), "next")
}

// onGPU returns a kernel without a call, on the GPU the process names by index.
func onGPU(pid libpf.PID, index uint64, duration uint64) event {
	ev := kernel(0, duration)
	ev.pid = pid
	ev.gpuIndex = index
	return ev
}

// gpuOf returns the GPU attribute of a sample, or "".
func gpuOf(s samples.SnapshotSample) string {
	for _, attr := range s.Attributes {
		if attr.Key == hwIDKey {
			return attr.Value.AsString()
		}
	}
	return ""
}

func TestMatcherGPUIndexes(t *testing.T) {
	m := newTestMatcher(t, liveProcesses{testPID: "/bin/app", testPID + 1: "/bin/other"})
	now := time.Now()

	// Work waits for the UUID of its GPU, as when the DEVICE event was lost or
	// went to another profiler.
	addKernel(t, m, onGPU(testPID, 1, 10), "early", now)
	assert.Empty(t, snapshotSamples(t, m, now))
	m.setGPU(testPID, 1, "GPU-a", now)
	addKernel(t, m, onGPU(testPID, 1, 20), "late", now)
	got := snapshotSamples(t, m, now)
	assert.Equal(t, "GPU-a", gpuOf(got["early (no stack)"]))
	assert.Equal(t, int64(10), got["early (no stack)"].Value)
	assert.Equal(t, "GPU-a", gpuOf(got["late (no stack)"]))

	// Indexes belong to their process.
	addKernel(t, m, onGPU(testPID+1, 1, 30), "other", now)
	assert.Empty(t, snapshotSamples(t, m, now))

	// Work whose GPU never comes is reported without one.
	got = snapshotSamples(t, m, now.Add(gpuGrace))
	assert.Empty(t, gpuOf(got["other (no stack)"]))
	assert.Equal(t, int64(30), got["other (no stack)"].Value)

	// The work of a process that exits waits only until it is retired.
	addKernel(t, m, onGPU(testPID, 2, 50), "exited", now)
	m.processExited(testPID, now)
	assert.Empty(t, snapshotSamples(t, m, now))
	got = snapshotSamples(t, m, now.Add(exitGrace))
	assert.Empty(t, gpuOf(got["exited (no stack)"]))
	assert.Equal(t, int64(50), got["exited (no stack)"].Value)

	// A process reusing the PID starts with no indexes.
	m.processExited(testPID, now)
	m.processStarted(testPID)
	addKernel(t, m, onGPU(testPID, 1, 40), "reused", now)
	assert.Empty(t, snapshotSamples(t, m, now))
	m.setGPU(testPID, 1, "GPU-b", now)
	assert.Equal(t, "GPU-b", gpuOf(snapshotSamples(t, m, now)["reused (no stack)"]))
}

func TestMatcherBoundsUnresolvedWork(t *testing.T) {
	m := newTestMatcher(t, liveProcesses{testPID: "/bin/app"})
	now := time.Now()
	for range unresolvedCapacity {
		addKernel(t, m, onGPU(testPID, 1, 1), "waiting", now)
	}
	// Beyond the bound, work is reported at once, without a GPU.
	addKernel(t, m, onGPU(testPID, 1, 5), "beyond", now)
	got := snapshotSamples(t, m, now)
	assert.Len(t, got, 1)
	assert.Empty(t, gpuOf(got["beyond (no stack)"]))

	m.setGPU(testPID, 1, "GPU-a", now)
	got = snapshotSamples(t, m, now)
	assert.Equal(t, int64(unresolvedCapacity), got["waiting (no stack)"].Value)
	assert.Equal(t, "GPU-a", gpuOf(got["waiting (no stack)"]))
}
