// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package gpu

import (
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

func kernel(correlationID uint32, duration uint64) kernelRecord {
	return kernelRecord{pid: testPID, start: 10, end: 10 + duration, correlationID: correlationID}
}

// addKernel adds a kernel execution to m.
func addKernel(m *matcher, rec kernelRecord, name string, now time.Time) {
	m.addWork(kernelWork(rec, name, ""), now)
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
	m, err := newMatcher(live.meta, newNameCache())
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
			require.Equal(t, libpf.GPUFrame, leaf.Type)
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
	addKernel(m, kernel(1, 100), "vector_add", now)
	// Arrives before its launch.
	addKernel(m, kernel(2, 50), "vector_add", now)
	m.addLaunch(testPID, 2, hostTrace(0x10, 0x20), now)
	m.addLaunch(testPID, 3, hostTrace(0x30), now)
	addKernel(m, kernel(3, 7), "reduce", now)

	got := snapshotSamples(t, m, now)
	require.Len(t, got, 2)
	assert.Equal(t, int64(150), got["vector_add"].Value)
	assert.Len(t, got["vector_add"].Frames, 3)
	assert.Equal(t, int64(7), got["reduce"].Value)
	assert.NotEqual(t, got["vector_add"].TraceHash, got["reduce"].TraceHash)
	assert.Equal(t, "/bin/app", got["reduce"].ProcessMeta.ExecutablePath.String())

	assert.Empty(t, m.snapshot(now), "samples are reported once")
}

func TestMatcherSameStackDifferentKernels(t *testing.T) {
	m := newTestMatcher(t, liveProcesses{testPID: "/bin/app"})
	now := time.Now()
	m.addLaunch(testPID, 1, hostTrace(0x10), now)
	addKernel(m, kernel(1, 1), "a", now)
	m.addLaunch(testPID, 2, hostTrace(0x10), now)
	addKernel(m, kernel(2, 1), "b", now)

	got := snapshotSamples(t, m, now)
	require.Len(t, got, 2)
	assert.NotEqual(t, got["a"].TraceHash, got["b"].TraceHash)
}

func TestMatcherGraphLaunch(t *testing.T) {
	m := newTestMatcher(t, liveProcesses{testPID: "/bin/app"})
	now := time.Now()
	graphKernel := kernel(5, 10)
	graphKernel.graphID = 9

	addKernel(m, graphKernel, "step", now)
	m.addLaunch(testPID, 5, hostTrace(0x10), now)
	addKernel(m, graphKernel, "step", now)
	addKernel(m, graphKernel, "step", now)

	got := snapshotSamples(t, m, now)
	require.Len(t, got, 1)
	assert.Equal(t, int64(30), got["step"].Value)
}

func TestMatcherUnmatchedKernelsExpire(t *testing.T) {
	m := newTestMatcher(t, liveProcesses{testPID: "/bin/app"})
	now := time.Now()
	addKernel(m, kernel(1, 40), "orphan", now)

	assert.Empty(t, m.snapshot(now), "kernel waits for its launch")
	got := snapshotSamples(t, m, now.Add(matchGrace))
	require.Len(t, got, 1)
	assert.Equal(t, int64(40), got["orphan (no stack)"].Value)
}

func TestMatcherInvalidKernel(t *testing.T) {
	m := newTestMatcher(t, liveProcesses{testPID: "/bin/app"})
	now := time.Now()
	m.addLaunch(testPID, 1, hostTrace(0x10), now)
	addKernel(m, kernelRecord{pid: testPID, start: 20, end: 20, correlationID: 1}, "k", now)
	assert.Empty(t, m.snapshot(now.Add(matchGrace)))
}

func TestMatcherReusedPID(t *testing.T) {
	m := newTestMatcher(t, liveProcesses{testPID: "/bin/app"})
	now := time.Now()
	m.addLaunch(testPID, 1, hostTrace(0x10), now)
	m.processExited(testPID, now)

	// A new process reuses the PID and the correlation id after exitGrace.
	later := now.Add(exitGrace)
	addKernel(m, kernel(1, 10), "k", later)
	got := snapshotSamples(t, m, later.Add(matchGrace))
	require.Len(t, got, 1)
	_, ok := got["k (no stack)"]
	assert.True(t, ok, "must not match the previous process's launch")
}

func TestMatcherMatchesInFlightRecordsOfExitedProcess(t *testing.T) {
	m := newTestMatcher(t, liveProcesses{testPID: "/bin/app"})
	now := time.Now()
	addKernel(m, kernel(1, 10), "k", now)
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
	addKernel(m, kernel(1, 10), "k", now)

	// The process exits and the tracer forgets it before the report.
	m.processExited(testPID, now)
	delete(live, testPID)

	got := snapshotSamples(t, m, now)
	require.Len(t, got, 1)
	assert.Equal(t, "/bin/short-job", got["k"].ProcessMeta.ExecutablePath.String())
}

func TestMatcherUnknownKernelName(t *testing.T) {
	m := newTestMatcher(t, liveProcesses{testPID: "/bin/app"})
	now := time.Now()
	m.addLaunch(testPID, 1, hostTrace(0x10), now)
	addKernel(m, kernel(1, 1), "", now)
	_, ok := snapshotSamples(t, m, now)[unknownKernel]
	assert.True(t, ok)
}

func TestMatcherMemoryOperationsKeepLaunch(t *testing.T) {
	m := newTestMatcher(t, liveProcesses{testPID: "/bin/app"})
	now := time.Now()
	m.addLaunch(testPID, 1, hostTrace(0x10), now)
	// A graph's copy and kernel share the correlation id.
	m.addWork(memcpyWork(memcpyRecord{pid: testPID, correlationID: 1, end: 5, copyKind: 1,
		bytes: 64}, ""), now)
	graphKernel := kernel(1, 3)
	graphKernel.graphID = 2
	addKernel(m, graphKernel, "k", now)

	byType := make(map[string]int64)
	for _, profile := range m.snapshot(now) {
		for _, s := range profile.Samples {
			assert.Len(t, s.Frames, 2, "every value joins the launch stack")
			byType[profile.ProfileType.SampleType] += s.Value
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
	m.addWork(kernelWork(kernel(1, 5), "k", "aten::mm\x1faten::matmul\x1ftrain_step"), now)
	m.addLaunch(testPID, 2, hostTrace(0x10), now)
	m.addWork(kernelWork(kernel(2, 7), "k", ""), now)

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

func TestMatcherWorkWithoutLaunch(t *testing.T) {
	m := newTestMatcher(t, liveProcesses{testPID: "/bin/app"})
	now := time.Now()
	work, _ := eventWork(eventRecord{pid: testPID, kind: eventUVMHtoD, v1: 4096})
	m.addWork(work, now)

	profiles := m.snapshot(now)
	require.Len(t, profiles, 1, "reported without waiting for a launch")
	require.Len(t, profiles[0].Samples, 1)
	assert.Equal(t, int64(4096), profiles[0].Samples[0].Value)
	assert.Len(t, profiles[0].Samples[0].Frames, 1)
}

func TestMatcherMergesPendingGraphWork(t *testing.T) {
	m := newTestMatcher(t, liveProcesses{testPID: "/bin/app"})
	now := time.Now()
	graphKernel := kernel(7, 2)
	graphKernel.graphID = 3
	// A graph launch runs more kernels than fit apart in the pending work.
	for range pendingCapacity + 10 {
		addKernel(m, graphKernel, "step", now)
	}
	addKernel(m, graphKernel, "reduce", now)
	assert.Equal(t, 2, m.numPending, "kernels of one launch and name wait as one")

	m.addLaunch(testPID, 7, hostTrace(0x10), now)
	got := snapshotSamples(t, m, now)
	require.Len(t, got, 2)
	assert.Equal(t, int64(2*(pendingCapacity+10)), got["step"].Value)
	assert.Equal(t, int64(2), got["reduce"].Value)
	assert.Zero(t, m.numPending)
}
