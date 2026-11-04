//go:build integration && linux

// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package gpu_test

import (
	"context"
	"maps"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/metric/noop"

	"go.opentelemetry.io/ebpf-profiler/interpreter/interpreterconfig"
	"go.opentelemetry.io/ebpf-profiler/libpf"
	"go.opentelemetry.io/ebpf-profiler/metrics"
	"go.opentelemetry.io/ebpf-profiler/probes/gpu"
	"go.opentelemetry.io/ebpf-profiler/reporter/samples"
	"go.opentelemetry.io/ebpf-profiler/tracer"
)

// counted holds the values of the profiler's counters.
var counted = &recordingMeter{values: make(map[string]int64)}

func TestMain(m *testing.M) {
	// Lingering periodic callers of the tracer add metrics concurrently.
	metrics.Start(counted)
	os.Exit(m.Run())
}

// recordingMeter adds up the values of the counters it creates, by name.
type recordingMeter struct {
	noop.Meter
	mu     sync.Mutex
	values map[string]int64
}

func (m *recordingMeter) Int64Counter(name string, _ ...metric.Int64CounterOption,
) (metric.Int64Counter, error) {
	return &recordingCounter{meter: m, name: name}, nil
}

func (m *recordingMeter) value(name string) int64 {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.values[name]
}

type recordingCounter struct {
	noop.Int64Counter
	meter *recordingMeter
	name  string
}

func (c *recordingCounter) Add(_ context.Context, value int64, _ ...metric.AddOption) {
	c.meter.mu.Lock()
	defer c.meter.mu.Unlock()
	c.meter.values[c.name] += value
}

type testIntervals struct{}

func (testIntervals) MonitorInterval() time.Duration       { return time.Second }
func (testIntervals) TracePollInterval() time.Duration     { return 250 * time.Millisecond }
func (testIntervals) PIDCleanupInterval() time.Duration    { return time.Second }
func (testIntervals) ExecutableUnloadDelay() time.Duration { return time.Second }

type reportedEvent struct {
	trace *libpf.Trace
	meta  samples.TraceEventMeta
	// values are the values the event reports.
	values []int64
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
	// The context values alias a pooled trace: keep what they report instead.
	values := meta.ProfileType.AppendValues(nil, meta)
	kept := *meta
	kept.ContextValues = nil
	r.events = append(r.events, reportedEvent{trace: trace, meta: kept, values: values})
	return nil
}

func (r *recordingReporter) eventsOf(pid libpf.PID, sampleType string) []reportedEvent {
	r.mu.Lock()
	defer r.mu.Unlock()
	var events []reportedEvent
	for _, e := range r.events {
		if e.meta.PID == pid && e.meta.ProfileType.SampleTypes[0].Type == sampleType {
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
		ProbesEnabled:          true,
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

// produced is what the fake producer fired, values multiplied by the sample
// weight of their launch.
type produced struct {
	pid           libpf.PID
	launches      uint64
	launchWeights int64
	kernels       uint64
	kernelNs      int64
	memNs         int64
	memBytes      int64
	// broken is the number of calls whose broken events the profiler drops.
	broken int64
}

// profiles returns the totals the profiler must report for what was produced.
func (p produced) profiles() map[string]int64 {
	return map[string]int64{
		"gpu_kernel_time": p.kernelNs,
		"gpu_mem_time":    p.memNs,
		"gpu_mem_bytes":   p.memBytes,
	}
}

// runProducer builds the fake producer and runs it: it waits for the probe to
// raise the semaphore of its USDT probe, then fires events for the given time.
func runProducer(t *testing.T, seconds int) produced {
	t.Helper()
	return runBuiltProducer(t, buildProducer(t), seconds)
}

// runBuiltProducer runs the fake producer built by buildProducer.
func runBuiltProducer(t *testing.T, bin string, seconds int) produced {
	t.Helper()
	cmd := exec.Command(bin, "10", strconv.Itoa(seconds))
	var stdout, stderr strings.Builder
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	require.NoError(t, cmd.Start())
	require.NoError(t, cmd.Wait(), stderr.String())

	fields := strings.Fields(stdout.String())
	require.Len(t, fields, 7, stdout.String())
	values := make([]int64, len(fields))
	for i, field := range fields {
		var err error
		values[i], err = strconv.ParseInt(field, 10, 64)
		require.NoError(t, err)
	}
	require.NotZero(t, values[0])
	return produced{
		pid:           libpf.PID(cmd.Process.Pid),
		launches:      uint64(values[0]),
		launchWeights: values[1],
		kernels:       uint64(values[2]),
		kernelNs:      values[3],
		memNs:         values[4],
		memBytes:      values[5],
		broken:        values[6],
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
		weights += e.values[0]

		require.True(t, inProducer(t, e.trace.Frames), "launch stack misses the producer")
		for _, handle := range e.trace.Frames {
			require.NotEqual(t, libpf.KernelFrame, handle.Value().Type)
		}
	}
	require.Equal(t, fired.launchWeights, weights)
}

// testGPU is the GPU UUID the fake producer names by an index in its events.
const testGPU = "GPU-00000000-0000-0000-0000-000000000001"

// deviceLaunched is the kernel the fake producer reports without a call.
const deviceLaunched = "device_launched"

// collectGPUProfiles collects the snapshot samples of pid until the totals by
// profile type reach want. It returns the totals by profile type, then by leaf
// or by the GPU frames above the host stack.
func collectGPUProfiles(t *testing.T, trc *tracer.Tracer, pid libpf.PID,
	want map[string]int64,
) (byLeaf, byStack map[string]map[string]int64) {
	t.Helper()
	totals := make(map[string]int64)
	byLeaf = make(map[string]map[string]int64)
	byStack = make(map[string]map[string]int64)
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		time.Sleep(500 * time.Millisecond)
		for _, profile := range trc.CollectSnapshots() {
			sampleType := profile.ProfileType.SampleTypes[0].Type
			for _, sample := range profile.Samples {
				if sample.PID != pid {
					continue
				}
				leaf := sample.Frames[0].Value()
				require.Equal(t, libpf.CUDAFrame, leaf.Type)
				// Kernels that no call started have no launch stack.
				require.Equal(t, leaf.FunctionName.String() != deviceLaunched,
					inProducer(t, sample.Frames[1:]),
					"launch stack of %s %s", sampleType, leaf.FunctionName)
				require.NotNil(t, sample.ProcessMeta)
				require.Equal(t, "producer",
					filepath.Base(sample.ProcessMeta.ExecutablePath.String()))
				require.Equal(t, []attribute.KeyValue{
					attribute.String("hw.id", testGPU),
					attribute.String("hw.type", "gpu"),
				}, sample.Attributes)
				if byLeaf[sampleType] == nil {
					byLeaf[sampleType] = make(map[string]int64)
					byStack[sampleType] = make(map[string]int64)
				}
				byLeaf[sampleType][leaf.FunctionName.String()] += sample.Value
				byStack[sampleType][gpuFrames(sample.Frames)] += sample.Value
				totals[sampleType] += sample.Value
			}
		}
		if maps.Equal(totals, want) {
			break
		}
	}
	require.Equal(t, want, totals)
	return byLeaf, byStack
}

// gpuFrames joins the names of the GPU frames of a sample.
func gpuFrames(frames libpf.Frames) string {
	var names []string
	for _, handle := range frames {
		frame := handle.Value()
		if frame.Type != libpf.CUDAFrame {
			break
		}
		names = append(names, frame.FunctionName.String())
	}
	return strings.Join(names, " > ")
}

func TestGPUProfiles(t *testing.T) {
	rep := &recordingReporter{}
	trc, _ := startTracer(t, rep, gpu.Config{})
	// The metrics of a second are reported in the next one, so those of an
	// earlier test may still be pending.
	time.Sleep(2 * testIntervals{}.MonitorInterval())
	tooLarge := counted.value("bpf.errors.gpu_event_too_large")
	unreadable := counted.value("bpf.errors.gpu_event_read")
	fired := runProducer(t, 3)

	// The totals leave out the events the producer breaks on purpose.
	byLeaf, byStack := collectGPUProfiles(t, trc, fired.pid, fired.profiles())
	require.Equal(t, []string{deviceLaunched, "graph_step", "vector_add"},
		slices.Sorted(maps.Keys(byLeaf["gpu_kernel_time"])))
	require.Equal(t, []string{"memcpy:HtoD", "memset"},
		slices.Sorted(maps.Keys(byLeaf["gpu_mem_time"])))
	require.Equal(t, []string{
		deviceLaunched,
		"graph_step",
		"vector_add",
		"vector_add > range:aten::mm > range:aten::matmul > range:train_step",
	}, slices.Sorted(maps.Keys(byStack["gpu_kernel_time"])))
	require.Equal(t, []string{"memcpy:HtoD > range:train_step", "memset > range:train_step"},
		slices.Sorted(maps.Keys(byStack["gpu_mem_time"])))

	// Each call with broken events sent one too large and one unreadable.
	require.EventuallyWithT(t, func(c *assert.CollectT) {
		assert.Equal(c, fired.broken,
			counted.value("bpf.errors.gpu_event_too_large")-tooLarge, "too large")
		assert.Equal(c, fired.broken,
			counted.value("bpf.errors.gpu_event_read")-unreadable, "unreadable")
	}, 5*time.Second, 100*time.Millisecond)
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

// TestProcessesShareUprobes runs the same producer file in several processes,
// one after the other and two at once: the uprobes attached for the first are
// reused for the others. Processes that start while the uprobes are attached
// fire before the profiler knows them, so only the first one has the stacks of
// all its launches.
func TestProcessesShareUprobes(t *testing.T) {
	rep := &recordingReporter{}
	trc, _ := startTracer(t, rep, gpu.Config{})
	bin := buildProducer(t)

	first := runBuiltProducer(t, bin, 1)
	var both [2]produced
	var wg sync.WaitGroup
	for i := range both {
		wg.Go(func() { both[i] = runBuiltProducer(t, bin, 1) })
	}
	wg.Wait()

	want := make(map[libpf.PID]map[string]int64)
	for _, fired := range []produced{first, both[0], both[1]} {
		want[fired.pid] = fired.profiles()
	}
	got := make(map[libpf.PID]map[string]int64)
	// stacked and stackless add up the values of the samples of each process
	// below their launch stack, and without one.
	stacked := make(map[libpf.PID]int64)
	stackless := make(map[libpf.PID]int64)
	require.Eventually(t, func() bool {
		for _, profile := range trc.CollectSnapshots() {
			for _, sample := range profile.Samples {
				if _, ok := want[sample.PID]; !ok {
					continue
				}
				if got[sample.PID] == nil {
					got[sample.PID] = make(map[string]int64)
				}
				got[sample.PID][profile.ProfileType.SampleTypes[0].Type] += sample.Value
				if sample.Frames[0].Value().FunctionName.String() == deviceLaunched {
					continue
				}
				if inProducer(t, sample.Frames[1:]) {
					stacked[sample.PID] += sample.Value
				} else {
					stackless[sample.PID] += sample.Value
				}
			}
		}
		return maps.EqualFunc(want, got, maps.Equal)
	}, 10*time.Second, 500*time.Millisecond, "want %v, got %v", want, got)
	require.Zero(t, stackless[first.pid], "GPU work of the first process without its launch stack")

	for i, fired := range []produced{first, both[0], both[1]} {
		require.NotZero(t, stacked[fired.pid], "no GPU work of %d below its launch", fired.pid)
		var weights int64
		require.Eventually(t, func() bool {
			weights = 0
			for _, e := range rep.eventsOf(fired.pid, "gpu_launches") {
				require.True(t, inProducer(t, e.trace.Frames), "launch stack misses the producer")
				weights += e.values[0]
			}
			if i == 0 {
				return weights == fired.launchWeights
			}
			return weights > 0
		}, 5*time.Second, 100*time.Millisecond,
			"launch weights of %d: %d of %d", fired.pid, weights, fired.launchWeights)
		require.LessOrEqual(t, weights, fired.launchWeights)
	}
}

// TestProducerPaths attaches to a producer given by path as the probe loads,
// before any process maps it: the semaphore is up as soon as the producer
// starts, and its work is reported.
func TestProducerPaths(t *testing.T) {
	bin := buildProducer(t)
	rep := &recordingReporter{}
	trc, _ := startTracer(t, rep, gpu.Config{ProducerPaths: []string{bin}})
	fired := runBuiltProducer(t, bin, 2)

	want := fired.profiles()
	got := make(map[string]int64)
	require.Eventually(t, func() bool {
		for _, profile := range trc.CollectSnapshots() {
			for _, sample := range profile.Samples {
				if sample.PID == fired.pid {
					got[profile.ProfileType.SampleTypes[0].Type] += sample.Value
				}
			}
		}
		return maps.Equal(want, got)
	}, 10*time.Second, 500*time.Millisecond, "want %v, got %v", want, got)
	require.NotEmpty(t, rep.eventsOf(fired.pid, "gpu_launches"))
}
