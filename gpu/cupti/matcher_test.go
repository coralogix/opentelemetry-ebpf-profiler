// SPDX-License-Identifier: Apache-2.0

package cupti

import (
	"sync"
	"testing"

	"go.opentelemetry.io/ebpf-profiler/libpf"
	"go.opentelemetry.io/ebpf-profiler/reporter/samples"
)

// capRep captures reported traces for assertions.
type capRep struct {
	mu      sync.Mutex
	traces  []*libpf.Trace
	metas   []samples.TraceEventMeta
	failNum int // fail the first failNum reports
	calls   int
}

func (c *capRep) ReportTraceEvent(t *libpf.Trace, m *samples.TraceEventMeta) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.calls++
	if c.calls <= c.failNum {
		return errReport
	}
	c.traces = append(c.traces, t)
	c.metas = append(c.metas, *m)
	return nil
}

var errReport = &reportErr{}

type reportErr struct{}

func (*reportErr) Error() string { return "report failed" }

const (
	testOrigin         = libpf.Origin(0x42)
	testMemOrigin      = libpf.Origin(0x43)
	testMemBytesOrigin = libpf.Origin(0x44)
	testAPIOrigin      = libpf.Origin(0x45)
)

var testOrigins = Origins{
	KernelTime:   testOrigin,
	MemTime:      testMemOrigin,
	MemBytes:     testMemBytesOrigin,
	APITime:      testAPIOrigin,
	BusyTime:     libpf.Origin(0x46),
	UVMBytes:     libpf.Origin(0x47),
	UVMFaults:    libpf.Origin(0x48),
	AllocBytes:   libpf.Origin(0x49),
	StallSamples: libpf.Origin(0x4a),
	NCCLTime:     libpf.Origin(0x4b),
	NCCLBytes:    libpf.Origin(0x4c),
}

// hostFrame builds a host frame whose collapsed module name is the given string.
func hostFrame(mod string) libpf.Frames {
	var f libpf.Frames
	f.Append(&libpf.Frame{Type: libpf.NativeFrame, FunctionName: libpf.Intern(mod)})
	return f
}

func launch(m *Matcher, corr uint32, pid libpf.PID, mod string) {
	tr := &libpf.Trace{Frames: hostFrame(mod)}
	m.OnLaunchTrace(tr, &samples.TraceEventMeta{
		Origin: testOrigin,
		Value:  int64(corr), // GPU launches carry the correlation id in Value
		PID:    pid,
		Comm:   libpf.Intern("worker"),
	})
}

func TestMatcherEmitsGPUTimeOnFlush(t *testing.T) {
	rep := &capRep{}
	m := NewMatcher(rep, testOrigins, NewNameCache())

	launch(m, 1, 100, "libcuda.so")
	m.OnTiming(KernelTiming{CorrelationID: 1, Start: 1000, End: 4000, PID: 100})

	m.Flush()

	if len(rep.traces) != 1 {
		t.Fatalf("want 1 reported trace, got %d", len(rep.traces))
	}
	got := rep.metas[0]
	if got.Value != 3000 {
		t.Errorf("value = %d, want 3000 (GPU ns)", got.Value)
	}
	if got.Origin != testOrigin {
		t.Errorf("origin = %d, want %d", got.Origin, testOrigin)
	}
	if got.PID != 100 {
		t.Errorf("PID = %d, want 100", got.PID)
	}
	// Leaf frame is the synthetic GPU kernel frame.
	leaf := rep.traces[0].Frames[0].Value()
	if leaf.Type != libpf.GPUKernelFrame {
		t.Errorf("leaf type = %v, want GPUKernelFrame", leaf.Type)
	}
	if matched, _ := m.Stats(); matched != 1 {
		t.Errorf("matched = %d, want 1", matched)
	}
}

func TestMatcherFlushEmitsOnlyDelta(t *testing.T) {
	rep := &capRep{}
	m := NewMatcher(rep, testOrigins, NewNameCache())

	launch(m, 1, 100, "libcuda.so")
	m.OnTiming(KernelTiming{CorrelationID: 1, Start: 0, End: 1000, PID: 100})
	m.Flush() // emits 1000

	// No new timing → second flush emits nothing.
	m.Flush()
	if len(rep.traces) != 1 {
		t.Fatalf("second flush emitted again: %d traces", len(rep.traces))
	}

	// New launch+timing on same (pid, kernel, stack) → only the new delta.
	launch(m, 2, 100, "libcuda.so")
	m.OnTiming(KernelTiming{CorrelationID: 2, Start: 0, End: 500, PID: 100})
	m.Flush()
	if len(rep.traces) != 2 {
		t.Fatalf("want 2 emits, got %d", len(rep.traces))
	}
	if rep.metas[1].Value != 500 {
		t.Errorf("delta value = %d, want 500", rep.metas[1].Value)
	}
}

func TestMatcherUnmatchedFallsBack(t *testing.T) {
	rep := &capRep{}
	m := NewMatcher(rep, testOrigins, NewNameCache())

	// Timing with no buffered launch: the duration still counts (kernel time
	// is the authoritative total), attributed to the process only.
	m.OnTiming(KernelTiming{CorrelationID: 99, Start: 0, End: 1000, PID: 1})
	m.Flush()
	if len(rep.traces) != 1 {
		t.Fatalf("want 1 fallback trace, got %d", len(rep.traces))
	}
	if rep.metas[0].Value != 1000 {
		t.Errorf("value = %d, want 1000", rep.metas[0].Value)
	}
	if n := len(rep.traces[0].Frames); n != 1 {
		t.Errorf("fallback trace frames = %d, want 1 (leaf only)", n)
	}
	if _, unmatched := m.Stats(); unmatched != 1 {
		t.Errorf("unmatched = %d, want 1", unmatched)
	}
}

func TestMatcherRetriesOnReportError(t *testing.T) {
	rep := &capRep{failNum: 1} // first report fails
	m := NewMatcher(rep, testOrigins, NewNameCache())

	launch(m, 1, 100, "libcuda.so")
	m.OnTiming(KernelTiming{CorrelationID: 1, Start: 0, End: 2000, PID: 100})

	m.Flush() // fails, delta NOT advanced
	if len(rep.traces) != 0 {
		t.Fatalf("expected first emit to fail")
	}
	m.Flush() // retry succeeds with full delta
	if len(rep.traces) != 1 || rep.metas[0].Value != 2000 {
		t.Fatalf("retry did not re-emit full delta: %+v", rep.metas)
	}
}

func TestMatcherPruneDeadPIDs(t *testing.T) {
	rep := &capRep{}
	m := NewMatcher(rep, testOrigins, NewNameCache())

	launch(m, 1, 100, "libcuda.so")
	m.OnTiming(KernelTiming{CorrelationID: 1, Start: 0, End: 1000, PID: 100})
	m.Flush()

	m.PruneDeadPIDs([]int{100})

	// A fresh launch+timing for the same pid after prune starts a new bucket;
	// its delta is the new kernel's full time (not affected by the pruned one).
	launch(m, 2, 100, "libcuda.so")
	m.OnTiming(KernelTiming{CorrelationID: 2, Start: 0, End: 700, PID: 100})
	m.Flush()
	if rep.metas[len(rep.metas)-1].Value != 700 {
		t.Errorf("post-prune delta = %d, want 700", rep.metas[len(rep.metas)-1].Value)
	}
}

func TestMatcherEmitsMemTimeOnFlush(t *testing.T) {
	rep := &capRep{}
	m := NewMatcher(rep, testOrigins, NewNameCache())

	// Two HtoD copies + one DtoH for the same pid, no buffered launch
	// (unmatched fallback) → two buckets, each reported twice (time + bytes).
	m.OnMem(MemTiming{Start: 0, End: 1000, CopyKind: 1, Bytes: 4096, PID: 100})
	m.OnMem(MemTiming{Start: 2000, End: 2500, CopyKind: 1, Bytes: 4096, PID: 100})
	m.OnMem(MemTiming{Start: 3000, End: 3300, CopyKind: 2, Bytes: 64, PID: 100})

	m.Flush()

	if len(rep.traces) != 4 {
		t.Fatalf("want 4 mem traces (2 buckets x time+bytes), got %d", len(rep.traces))
	}
	timeVals := map[int64]bool{}
	byteVals := map[int64]bool{}
	for _, mt := range rep.metas {
		switch mt.Origin {
		case testMemOrigin:
			timeVals[mt.Value] = true
		case testMemBytesOrigin:
			byteVals[mt.Value] = true
		default:
			t.Errorf("unexpected origin %d", mt.Origin)
		}
		if mt.PID != 100 {
			t.Errorf("PID = %d, want 100", mt.PID)
		}
	}
	if !timeVals[1500] || !timeVals[300] { // HtoD 1000+500, DtoH 300
		t.Errorf("missing time deltas, got %+v", timeVals)
	}
	if !byteVals[8192] || !byteVals[64] { // HtoD 4096+4096, DtoH 64
		t.Errorf("missing byte deltas, got %+v", byteVals)
	}
	// Unmatched copies: single leaf frame, no host stack.
	for _, tr := range rep.traces {
		if len(tr.Frames) != 1 {
			t.Errorf("mem trace frames = %d, want 1 (leaf only)", len(tr.Frames))
		}
	}

	// Delta semantics: nothing new → nothing emitted.
	m.Flush()
	if len(rep.traces) != 4 {
		t.Fatalf("second flush re-emitted: %d traces", len(rep.traces))
	}

	// Prune drops the buckets.
	m.PruneDeadPIDs([]int{100})
	m.OnMem(MemTiming{Start: 0, End: 200, CopyKind: 1, PID: 100})
	m.Flush()
	if got := rep.metas[len(rep.metas)-1].Value; got != 200 {
		t.Errorf("post-prune mem delta = %d, want 200", got)
	}
}

func TestMatcherMemJoinsHostStack(t *testing.T) {
	rep := &capRep{}
	m := NewMatcher(rep, testOrigins, NewNameCache())

	// The shim tracks memcpy call sites like launches: a buffered host stack
	// with the copy's correlation id attributes the copy to its host path.
	launch(m, 7, 100, "libcudart.so")
	m.OnMem(MemTiming{CorrelationID: 7, Start: 0, End: 800, CopyKind: 1, Bytes: 1024, PID: 100})

	m.Flush()

	if len(rep.traces) != 2 { // time + bytes
		t.Fatalf("want 2 traces, got %d", len(rep.traces))
	}
	for i, tr := range rep.traces {
		if len(tr.Frames) != 2 {
			t.Fatalf("trace %d frames = %d, want 2 (leaf + host module)", i, len(tr.Frames))
		}
		leaf := tr.Frames[0].Value()
		if leaf.FunctionName.String() != "memcpy:HtoD" {
			t.Errorf("leaf = %q, want memcpy:HtoD", leaf.FunctionName.String())
		}
		host := tr.Frames[1].Value()
		if host.FunctionName.String() != "libcudart.so" {
			t.Errorf("host frame = %q, want libcudart.so", host.FunctionName.String())
		}
	}
	// meta carries the launching process identity from the buffered launch.
	for _, mt := range rep.metas {
		if mt.Comm.String() != "worker" {
			t.Errorf("comm = %q, want worker (from launch meta)", mt.Comm.String())
		}
	}
}

func TestMatcherCorrIsPerProcess(t *testing.T) {
	rep := &capRep{}
	m := NewMatcher(rep, testOrigins, NewNameCache())

	// CUPTI correlation ids are per-process counters: the same id from two
	// processes must not cross-match.
	launch(m, 5, 100, "libcuda.so")
	m.OnTiming(KernelTiming{CorrelationID: 5, Start: 0, End: 1000, PID: 200})
	if matched, unmatched := m.Stats(); matched != 0 || unmatched != 1 {
		t.Fatalf("cross-process timing matched: matched=%d unmatched=%d", matched, unmatched)
	}
	m.OnTiming(KernelTiming{CorrelationID: 5, Start: 0, End: 1000, PID: 100})
	if matched, _ := m.Stats(); matched != 1 {
		t.Fatalf("same-process timing did not match")
	}
}

func TestMatcherOnEventBuckets(t *testing.T) {
	rep := &capRep{}
	m := NewMatcher(rep, testOrigins, NewNameCache())

	// API span (cookie 0 = first apiSymbols entry) and UVM bytes; both are
	// fallback-attributed (no correlation id).
	m.OnEvent(GPUEvent{Kind: EvAPISpan, Start: 0, End: 5000, V1: 0, PID: 100})
	m.OnEvent(GPUEvent{Kind: EvUVMHtoD, V1: 4096, PID: 100})
	m.Flush()

	if len(rep.traces) != 2 {
		t.Fatalf("want 2 traces (api + uvm), got %d", len(rep.traces))
	}
	byOrigin := map[libpf.Origin]int64{}
	leaves := map[string]bool{}
	for i, mt := range rep.metas {
		byOrigin[mt.Origin] = mt.Value
		leaves[rep.traces[i].Frames[0].Value().FunctionName.String()] = true
	}
	if byOrigin[testOrigins.APITime] != 5000 {
		t.Errorf("api time = %d, want 5000", byOrigin[testOrigins.APITime])
	}
	if byOrigin[testOrigins.UVMBytes] != 4096 {
		t.Errorf("uvm bytes = %d, want 4096", byOrigin[testOrigins.UVMBytes])
	}
	if !leaves[apiSymbols[0].display] || !leaves["uvm:HtoD"] {
		t.Errorf("unexpected leaves: %v", leaves)
	}
}

func TestMatcherMemsetViaEvent(t *testing.T) {
	rep := &capRep{}
	m := NewMatcher(rep, testOrigins, NewNameCache())

	launch(m, 9, 100, "libcudart.so")
	m.OnEvent(GPUEvent{Kind: EvMemset, Start: 0, End: 700, CorrelationID: 9,
		V1: 512, PID: 100})
	m.Flush()

	if len(rep.traces) != 2 { // time + bytes
		t.Fatalf("want 2 traces, got %d", len(rep.traces))
	}
	if got := rep.traces[0].Frames[0].Value().FunctionName.String(); got != "memset" {
		t.Errorf("leaf = %q, want memset", got)
	}
	if len(rep.traces[0].Frames) != 2 {
		t.Errorf("memset trace frames = %d, want 2 (leaf + host)", len(rep.traces[0].Frames))
	}
}

func TestCopyKindName(t *testing.T) {
	cases := map[uint32]string{0: "memcpy:unknown", 1: "memcpy:HtoD", 2: "memcpy:DtoH",
		8: "memcpy:DtoD", 99: "memcpy:unknown", copyKindMemset: "memset"}
	for in, want := range cases {
		if got := copyKindName(in); got != want {
			t.Errorf("copyKindName(%d) = %q, want %q", in, got, want)
		}
	}
}

func TestDemangleKernel(t *testing.T) {
	cases := map[string]string{
		"_Z12rms_norm_f32ILi1024ELb1ELb0EEvPKfPfi": "rms_norm_f32",
		"rms_norm_f32": "rms_norm_f32", // already clean
		"":             "",
	}
	for in, want := range cases {
		if got := demangleKernel(in); got != want {
			t.Errorf("demangleKernel(%q) = %q, want %q", in, got, want)
		}
	}
}
