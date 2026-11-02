// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package gpu // import "go.opentelemetry.io/ebpf-profiler/probes/gpu"

import (
	"math"
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
		SampleTypes:       []samples.SampleType{{Type: sampleType, Unit: unit}},
		ReportValues:      true,
		OmitThreadContext: true,
	}
}

// unknownKernel names kernels without a name.
const unknownKernel = "[unknown kernel]"

// int64Value converts a payload number, which the producer controls, to a
// profile value; values out of range count as 0.
func int64Value(v uint64) int64 {
	if v > math.MaxInt64 {
		return 0
	}
	return int64(v)
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
}

// scaled multiplies the work's values by the sample weight of its call, the
// number of calls it stands for. A weight of 0 counts as 1.
func (w gpuWork) scaled() gpuWork {
	for i := range w.values {
		w.values[i].value *= int64(max(w.sampleWeight, 1))
	}
	return w
}

// eventWork translates an event into GPU work; ok is false for kinds the probe
// does not know. Kernel names must already be demangled.
func eventWork(ev event) (work gpuWork, ok bool) {
	work = gpuWork{
		pid:           ev.pid,
		correlationID: uint32(ev.correlationID),
		sampleWeight:  ev.sampleWeight,
		ranges:        ev.ranges,
		gpu:           ev.gpu,
	}
	switch ev.kind {
	case kindKernel:
		work.leaf = ev.name
		if work.leaf == "" {
			work.leaf = unknownKernel
		}
		return work.withValue(kernelTimeProfileType, int64Value(ev.durationNS)), true
	case kindCopy:
		direction := ev.name
		if direction == "" {
			direction = "unknown"
		}
		work.leaf = "memcpy:" + direction
	case kindFill:
		work.leaf = "memset"
	default:
		return gpuWork{}, false
	}
	return work.withValue(memTimeProfileType, int64Value(ev.durationNS)).
		withValue(memBytesProfileType, int64Value(ev.bytes)), true
}
