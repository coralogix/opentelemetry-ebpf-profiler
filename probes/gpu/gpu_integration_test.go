//go:build integration && linux

// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package gpu_test

import (
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel/metric/noop"

	"go.opentelemetry.io/ebpf-profiler/interpreter/interpreterconfig"
	"go.opentelemetry.io/ebpf-profiler/libpf"
	"go.opentelemetry.io/ebpf-profiler/metrics"
	"go.opentelemetry.io/ebpf-profiler/probes/gpu"
	"go.opentelemetry.io/ebpf-profiler/reporter/samples"
	"go.opentelemetry.io/ebpf-profiler/tracer"
)

func TestMain(m *testing.M) {
	// Lingering periodic callers of the tracer add metrics concurrently.
	metrics.Start(noop.Meter{})
	os.Exit(m.Run())
}

type testIntervals struct{}

func (testIntervals) MonitorInterval() time.Duration       { return time.Second }
func (testIntervals) TracePollInterval() time.Duration     { return 250 * time.Millisecond }
func (testIntervals) PIDCleanupInterval() time.Duration    { return time.Second }
func (testIntervals) ExecutableUnloadDelay() time.Duration { return time.Second }

type reportedEvent struct {
	trace *libpf.Trace
	meta  samples.TraceEventMeta
}

// recordingReporter keeps every reported trace event.
type recordingReporter struct {
	mu     sync.Mutex
	events []reportedEvent
}

func (r *recordingReporter) ReportTraceEvent(trace *libpf.Trace,
	meta *samples.TraceEventMeta) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.events = append(r.events, reportedEvent{trace: trace, meta: *meta})
	return nil
}

func (r *recordingReporter) eventsOf(pid libpf.PID, sampleType string) []reportedEvent {
	r.mu.Lock()
	defer r.mu.Unlock()
	var events []reportedEvent
	for _, e := range r.events {
		if e.meta.PID == pid && e.meta.ProfileType.SampleType == sampleType {
			events = append(events, e)
		}
	}
	return events
}

// startTracer starts a tracer whose traces flow to rep, with the gpu probe
// enabled.
func startTracer(t *testing.T, rep *recordingReporter, config gpu.Config,
) (*tracer.Tracer, tracer.Probe) {
	t.Helper()
	ctx := t.Context()

	trc, err := tracer.NewTracer(ctx, &tracer.Config{
		TraceReporter:          rep,
		Intervals:              testIntervals{},
		InterpretersConfig:     interpreterconfig.AllInterpreters(),
		FilterErrorFrames:      true,
		SamplesPerSecond:       20,
		KernelVersionCheck:     true,
		ProbabilisticInterval:  100,
		ProbabilisticThreshold: 100,
	})
	require.NoError(t, err)
	t.Cleanup(trc.Close)

	trc.StartPIDEventProcessor(ctx)
	traces := make(chan *libpf.EbpfTrace, 64)
	require.NoError(t, trc.StartMapMonitors(ctx, traces))
	go func() {
		for {
			select {
			case trace := <-traces:
				if trace != nil {
					trc.HandleTrace(trace)
				}
			case <-ctx.Done():
				return
			}
		}
	}()
	require.NoError(t, trc.AttachSchedMonitor())

	probe, err := gpu.New(config)
	require.NoError(t, err)
	require.NoError(t, trc.Enable(ctx, probe))
	t.Cleanup(func() { require.NoError(t, probe.Unload()) })
	return trc, probe
}

// produced is what the fake producer fired, values multiplied by their sample
// weights.
type produced struct {
	pid             libpf.PID
	launches        uint64
	launchWeights   int64
	kernels         uint64
	kernelNs        int64
	memNs           int64
	memBytes        int64
	allocBytes      int64
	uvmBytes        int64
	stalls          int64
	collectiveNs    int64
	collectiveBytes int64
}

// runProducer builds the fake producer and runs it: it waits for the probe to
// raise the ready semaphore, then fires probes for the given time.
func runProducer(t *testing.T, seconds int) produced {
	t.Helper()
	bin := filepath.Join(t.TempDir(), "producer")
	out, err := exec.Command("cc", "-O2", "-g", "-o", bin, "testdata/producer.c").CombinedOutput()
	require.NoError(t, err, string(out))

	cmd := exec.Command(bin, "10", strconv.Itoa(seconds))
	var stdout, stderr strings.Builder
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	require.NoError(t, cmd.Start())
	require.NoError(t, cmd.Wait(), stderr.String())

	fields := strings.Fields(stdout.String())
	require.Len(t, fields, 11, stdout.String())
	values := make([]int64, len(fields))
	for i, field := range fields {
		values[i], err = strconv.ParseInt(field, 10, 64)
		require.NoError(t, err)
	}
	require.NotZero(t, values[0])
	return produced{
		pid:             libpf.PID(cmd.Process.Pid),
		launches:        uint64(values[0]),
		launchWeights:   values[1],
		kernels:         uint64(values[2]),
		kernelNs:        values[3],
		memNs:           values[4],
		memBytes:        values[5],
		allocBytes:      values[6],
		uvmBytes:        values[7],
		stalls:          values[8],
		collectiveNs:    values[9],
		collectiveBytes: values[10],
	}
}

func TestLaunchStacks(t *testing.T) {
	rep := &recordingReporter{}
	startTracer(t, rep, gpu.Config{})
	fired := runProducer(t, 3)

	var events []reportedEvent
	require.Eventually(t, func() bool {
		events = rep.eventsOf(fired.pid, "gpu_launches")
		return uint64(len(events)) == fired.launches
	}, 5*time.Second, 100*time.Millisecond,
		"got %d of %d launch stacks", len(events), fired.launches)

	var weights int64
	for _, e := range events {
		// The reported value is the sample weight the producer passed.
		weights += e.meta.Value

		require.True(t, inProducer(t, e.trace.Frames), "launch stack misses the producer")
		for _, handle := range e.trace.Frames {
			require.NotEqual(t, libpf.KernelFrame, handle.Value().Type)
		}
	}
	require.Equal(t, fired.launchWeights, weights)
}

// inProducer reports whether frames run code of the producer executable.
func inProducer(t *testing.T, frames libpf.Frames) bool {
	t.Helper()
	for _, handle := range frames {
		frame := handle.Value()
		if frame.Mapping.Valid() &&
			frame.Mapping.Value().File.Value().FileName.String() == "producer" {
			return true
		}
	}
	return false
}
