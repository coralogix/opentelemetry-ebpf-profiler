// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package gpu // import "go.opentelemetry.io/ebpf-profiler/probes/gpu"

import (
	"slices"

	"go.opentelemetry.io/ebpf-profiler/reporter/samples"
)

var (
	kernelTimeProfileType = gpuProfileType("gpu_kernel_time", "nanoseconds")
	memTimeProfileType    = gpuProfileType("gpu_mem_time", "nanoseconds")
	memBytesProfileType   = gpuProfileType("gpu_mem_bytes", "bytes")
)

// gpuProfileType describes a profile of GPU work. Its samples aggregate GPU
// activity, which runs on no CPU and in no thread.
func gpuProfileType(sampleType, unit string) *samples.TypeMetadata {
	return &samples.TypeMetadata{
		SampleType:        sampleType,
		SampleUnit:        unit,
		ReportValues:      true,
		OmitThreadContext: true,
	}
}

// unknownKernel names kernels whose name cannot be read.
const unknownKernel = "[unknown kernel]"

// copyKinds names the otel_gpu_copy_kind values.
var copyKinds = [...]string{1: "HtoD", 2: "DtoH", 3: "HtoA", 4: "AtoH", 5: "AtoA",
	6: "AtoD", 7: "DtoA", 8: "DtoD", 9: "HtoH", 10: "PtoP"}

func kindName(names []string, kind uint64) string {
	if kind < uint64(len(names)) && names[kind] != "" {
		return names[kind]
	}
	return "unknown"
}

// duration returns the length of a GPU activity, or 0 if the record is invalid.
func duration(start, end uint64) int64 {
	if end <= start {
		return 0
	}
	return int64(end - start)
}

// withValue appends value to the work's values if it is positive.
func (w gpuWork) withValue(profileType *samples.TypeMetadata, value int64) gpuWork {
	if value > 0 {
		w.values = append(w.values, profileValue{profileType: profileType, value: value})
	}
	return w
}

// merge adds the values of other, work of the same launch, to w's.
func (w *gpuWork) merge(other gpuWork) {
	for _, ov := range other.values {
		i := slices.IndexFunc(w.values, func(v profileValue) bool {
			return v.profileType == ov.profileType
		})
		if i < 0 {
			w.values = append(w.values, ov)
			continue
		}
		w.values[i].value += ov.value
	}
	w.consumesLaunch = w.consumesLaunch || other.consumesLaunch
}

// scaled multiplies the work's values by the sample weight of its call site,
// the number of call sites it stands for. A weight of 0 counts as 1.
func (w gpuWork) scaled(sampleWeight uint32) gpuWork {
	for i := range w.values {
		w.values[i].value *= int64(max(sampleWeight, 1))
	}
	return w
}

func kernelWork(rec kernelRecord, name string) gpuWork {
	if name == "" {
		name = unknownKernel
	}
	return gpuWork{
		pid:            rec.pid,
		correlationID:  rec.correlationID,
		leaf:           name,
		consumesLaunch: rec.graphID == 0,
	}.withValue(kernelTimeProfileType, duration(rec.start, rec.end)).scaled(rec.sampleWeight)
}

// memcpyWork is a memory copy. Its launch is kept: a CUDA graph shares one
// correlation id between its kernels and memory copies, and copy records do not
// say whether they ran in a graph.
func memcpyWork(rec memcpyRecord) gpuWork {
	return gpuWork{
		pid:           rec.pid,
		correlationID: rec.correlationID,
		leaf:          "memcpy:" + kindName(copyKinds[:], uint64(rec.copyKind)),
	}.withValue(memTimeProfileType, duration(rec.start, rec.end)).
		withValue(memBytesProfileType, int64(rec.bytes)).
		scaled(rec.sampleWeight)
}

// eventWork translates GPU events; ok is false for unknown kinds.
func eventWork(rec eventRecord) (work gpuWork, ok bool) {
	if rec.kind != eventMemset {
		return gpuWork{}, false
	}
	return gpuWork{
		pid:           rec.pid,
		correlationID: rec.correlationID,
		leaf:          "memset",
	}.withValue(memTimeProfileType, duration(rec.start, rec.end)).
		withValue(memBytesProfileType, int64(rec.v1)).
		scaled(rec.sampleWeight), true
}
