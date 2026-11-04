//go:build integration && linux

// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package gpu_test

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"go.opentelemetry.io/ebpf-profiler/libpf"
	"go.opentelemetry.io/ebpf-profiler/probes/gpu"
	"go.opentelemetry.io/ebpf-profiler/tracer"
)

// execGPU is the GPU the fake producer names by the same index after an exec.
const execGPU = "GPU-00000000-0000-0000-0000-000000000002"

// afterExec is the kernel the fake producer reports after an exec.
const afterExec = "after_exec"

// startModeProducer starts the fake producer in a mode that changes how it
// tells the profiler its GPU.
func startModeProducer(t *testing.T, bin string, seconds int, mode string,
) (cmd *exec.Cmd, stderr *strings.Builder) {
	t.Helper()
	cmd = exec.Command(bin, "10", strconv.Itoa(seconds), mode)
	stderr = &strings.Builder{}
	cmd.Stderr = stderr
	require.NoError(t, cmd.Start())
	return cmd, stderr
}

// gpusByLeaf collects the GPU samples of pid for a while, and returns their
// values by leaf, then by GPU: "" for samples without one.
func gpusByLeaf(trc *tracer.Tracer, pid libpf.PID, wait time.Duration,
) map[string]map[string]int64 {
	values := make(map[string]map[string]int64)
	deadline := time.Now().Add(wait)
	for time.Now().Before(deadline) {
		time.Sleep(500 * time.Millisecond)
		for _, profile := range trc.CollectSnapshots() {
			if profile.ProfileType.SampleTypes[0].Type != "gpu_kernel_time" {
				continue
			}
			for _, sample := range profile.Samples {
				if sample.PID != pid {
					continue
				}
				leaf := sample.Frames[0].Value().FunctionName.String()
				id := ""
				for _, attr := range sample.Attributes {
					if attr.Key == "hw.id" {
						id = attr.Value.AsString()
					}
				}
				if values[leaf] == nil {
					values[leaf] = make(map[string]int64)
				}
				values[leaf][id] += sample.Value
			}
		}
	}
	return values
}

// requireOnlyGPU checks that all kernel time of the leaves went to gpu.
func requireOnlyGPU(t *testing.T, values map[string]map[string]int64, gpu string,
	leaves ...string,
) {
	t.Helper()
	for _, leaf := range leaves {
		require.NotEmpty(t, values[leaf], "no samples of %s", leaf)
		for id, value := range values[leaf] {
			require.Equal(t, gpu, id, "%s: %d ns went to GPU %q, all by GPU: %v",
				leaf, value, id, values[leaf])
		}
	}
}

// TestGPUIndexLostDevice loses the first DEVICE event of a process, as when
// the ring buffer is full: its work must still get its GPU once the producer
// sends the mapping again.
func TestGPUIndexLostDevice(t *testing.T) {
	rep := &recordingReporter{}
	trc, _ := startTracer(t, rep, gpu.Config{})
	cmd, stderr := startModeProducer(t, buildProducer(t), 3, "lose-device")
	require.NoError(t, cmd.Wait(), stderr.String())

	values := gpusByLeaf(trc, libpf.PID(cmd.Process.Pid), 4*time.Second)
	requireOnlyGPU(t, values, testGPU, "vector_add", "graph_step")
}

// TestGPUIndexSecondProfiler starts a second profiler process while the first
// one runs: the producer cannot tell, as the semaphore stays up, and the second
// profiler must still get the GPU of all work it sees.
func TestGPUIndexSecondProfiler(t *testing.T) {
	first, _ := startTracer(t, &recordingReporter{}, gpu.Config{})
	cmd, stderr := startModeProducer(t, buildProducer(t), 8, "busy")
	time.Sleep(1500 * time.Millisecond)

	out := filepath.Join(t.TempDir(), "second.json")
	second := exec.Command(os.Args[0], "-test.run=^TestGPUIndexSecondProfilerHelper$")
	second.Env = append(os.Environ(), "GPU_TEST_PID="+strconv.Itoa(cmd.Process.Pid),
		"GPU_TEST_OUT="+out)
	var secondOutput strings.Builder
	second.Stdout = &secondOutput
	second.Stderr = &secondOutput
	require.NoError(t, second.Start())
	require.NoError(t, cmd.Wait(), stderr.String())
	t.Log(strings.TrimSpace(stderr.String()))

	pid := libpf.PID(cmd.Process.Pid)
	requireOnlyGPU(t, gpusByLeaf(first, pid, 3*time.Second), testGPU, "vector_add")
	require.NoError(t, second.Wait(), secondOutput.String())
	data, err := os.ReadFile(out)
	require.NoError(t, err)
	var values map[string]map[string]int64
	require.NoError(t, json.Unmarshal(data, &values))
	requireOnlyGPU(t, values, testGPU, "vector_add")
}

// TestGPUIndexSecondProfilerHelper is the second profiler of
// TestGPUIndexSecondProfiler, in a process of its own.
func TestGPUIndexSecondProfilerHelper(t *testing.T) {
	pid, err := strconv.Atoi(os.Getenv("GPU_TEST_PID"))
	if err != nil {
		t.Skip("run by TestGPUIndexSecondProfiler")
	}
	trc, _ := startTracer(t, &recordingReporter{}, gpu.Config{})
	// The producer started before this profiler: it finds it by sampling CPUs.
	require.NoError(t, trc.AttachTracer(nil))
	require.NoError(t, trc.EnableProfiling())
	data, err := json.Marshal(gpusByLeaf(trc, libpf.PID(pid), 8*time.Second))
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(os.Getenv("GPU_TEST_OUT"), data, 0o600))
}

// TestGPUIndexExec execs a producer that names another GPU by the same index:
// its work must not go to the GPU of the process before the exec.
func TestGPUIndexExec(t *testing.T) {
	rep := &recordingReporter{}
	trc, _ := startTracer(t, rep, gpu.Config{})
	cmd, stderr := startModeProducer(t, buildProducer(t), 4, "exec")
	require.NoError(t, cmd.Wait(), stderr.String())

	values := gpusByLeaf(trc, libpf.PID(cmd.Process.Pid), 4*time.Second)
	t.Logf("kernel time by leaf and GPU: %v", values)
	requireOnlyGPU(t, values, testGPU, "vector_add")
	requireOnlyGPU(t, values, execGPU, afterExec)
}
