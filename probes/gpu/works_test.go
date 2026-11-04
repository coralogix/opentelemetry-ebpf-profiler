// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package gpu

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"

	"go.opentelemetry.io/ebpf-profiler/libpf"
)

func TestKernelWork(t *testing.T) {
	work, ok := eventWork(event{pid: 1, correlationID: 1<<32 | 2, sampleWeight: 4,
		kind: kindKernel, runtime: runtimeCUDA, gpu: "GPU-1", name: "k", ranges: "r",
		durationNS: 20})
	assert.True(t, ok)
	assert.Equal(t, gpuWork{
		pid:           1,
		correlationID: 1<<32 | 2,
		sampleWeight:  4,
		frameType:     libpf.CUDAFrame,
		leaf:          "k",
		ranges:        "r",
		gpu:           "GPU-1",
		values:        []profileValue{{kernelTimeProfileType, 20}},
	}, work)

	unnamed, _ := eventWork(event{kind: kindKernel, durationNS: 20})
	assert.Equal(t, unknownKernel, unnamed.leaf)

	empty, _ := eventWork(event{kind: kindKernel, name: "k"})
	assert.Empty(t, empty.values)
}

func TestMemoryWork(t *testing.T) {
	for kind, leaf := range map[uint16]string{
		kindCopyHtoD: "memcpy:HtoD",
		kindCopyDtoH: "memcpy:DtoH",
		kindCopyDtoD: "memcpy:DtoD",
		kindCopyHtoH: "memcpy:HtoH",
	} {
		copyWork, ok := eventWork(event{pid: 1, kind: kind, durationNS: 50, bytes: 4096})
		assert.True(t, ok)
		assert.Equal(t, leaf, copyWork.leaf)
		assert.Equal(t, []profileValue{{memTimeProfileType, 50}, {memBytesProfileType, 4096}},
			copyWork.values)
	}

	fill, _ := eventWork(event{pid: 1, correlationID: 3, sampleWeight: 2, kind: kindFill,
		runtime: runtimeROCm, ranges: "r", durationNS: 4, bytes: 128})
	assert.Equal(t, gpuWork{
		pid:           1,
		correlationID: 3,
		sampleWeight:  2,
		frameType:     libpf.ROCmFrame,
		leaf:          "memset",
		ranges:        "r",
		values:        []profileValue{{memTimeProfileType, 4}, {memBytesProfileType, 128}},
	}, fill)
}

func TestNotWork(t *testing.T) {
	_, ok := eventWork(event{kind: 99, durationNS: 1})
	assert.False(t, ok, "unknown kind")
	_, ok = eventWork(event{kind: kindDevice, gpu: "GPU-1", gpuIndex: 1})
	assert.False(t, ok, "a GPU of the process")
}

func TestFrameType(t *testing.T) {
	assert.Equal(t, libpf.CUDAFrame, frameType(runtimeCUDA))
	assert.Equal(t, libpf.ROCmFrame, frameType(runtimeROCm))
	assert.Equal(t, libpf.UnknownFrame, frameType(0))
}

func TestOutOfRangeValues(t *testing.T) {
	work, _ := eventWork(event{kind: kindCopyHtoD, durationNS: math.MaxUint64, bytes: 8})
	assert.Equal(t, []profileValue{{memBytesProfileType, 8}}, work.values)
}

func TestWorkSampleWeight(t *testing.T) {
	kernel, _ := eventWork(event{kind: kindKernel, sampleWeight: 3, durationNS: 20})
	assert.Equal(t, []profileValue{{kernelTimeProfileType, 60}}, kernel.scaled().values)

	copyWork, _ := eventWork(event{kind: kindCopyDtoH, sampleWeight: 2, durationNS: 50,
		bytes: 4096})
	assert.Equal(t, []profileValue{{memTimeProfileType, 100}, {memBytesProfileType, 8192}},
		copyWork.scaled().values)

	// A weight of 0 counts as 1.
	unweighted, _ := eventWork(event{kind: kindKernel, durationNS: 20})
	assert.Equal(t, []profileValue{{kernelTimeProfileType, 20}}, unweighted.scaled().values)
}

func TestValuesSaturate(t *testing.T) {
	huge, _ := eventWork(event{kind: kindKernel, sampleWeight: math.MaxUint32,
		durationNS: math.MaxInt64 / 2})
	assert.Equal(t, []profileValue{{kernelTimeProfileType, math.MaxInt64}}, huge.scaled().values)

	huge.merge(huge)
	assert.Equal(t, []profileValue{{kernelTimeProfileType, math.MaxInt64}}, huge.values)
}
