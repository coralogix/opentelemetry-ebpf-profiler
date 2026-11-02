// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package gpu

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestKernelWork(t *testing.T) {
	work, ok := eventWork(event{pid: 1, correlationID: 1<<32 | 2, sampleWeight: 4,
		kind: kindKernel, gpu: "GPU-1", name: "k", ranges: "r", durationNS: 20})
	assert.True(t, ok)
	assert.Equal(t, gpuWork{
		pid:           1,
		correlationID: 2,
		sampleWeight:  4,
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
	copyWork, ok := eventWork(event{pid: 1, kind: kindCopy, name: "HtoD", durationNS: 50,
		bytes: 4096})
	assert.True(t, ok)
	assert.Equal(t, "memcpy:HtoD", copyWork.leaf)
	assert.Equal(t, []profileValue{{memTimeProfileType, 50}, {memBytesProfileType, 4096}},
		copyWork.values)

	unnamed, _ := eventWork(event{kind: kindCopy, durationNS: 1})
	assert.Equal(t, "memcpy:unknown", unnamed.leaf)

	fill, _ := eventWork(event{pid: 1, correlationID: 3, sampleWeight: 2, kind: kindFill,
		ranges: "r", durationNS: 4, bytes: 128})
	assert.Equal(t, gpuWork{
		pid:           1,
		correlationID: 3,
		sampleWeight:  2,
		leaf:          "memset",
		ranges:        "r",
		values:        []profileValue{{memTimeProfileType, 4}, {memBytesProfileType, 128}},
	}, fill)
}

func TestUnknownKind(t *testing.T) {
	_, ok := eventWork(event{kind: 99, durationNS: 1})
	assert.False(t, ok)
}

func TestOutOfRangeValues(t *testing.T) {
	work, _ := eventWork(event{kind: kindCopy, durationNS: math.MaxUint64, bytes: 8})
	assert.Equal(t, []profileValue{{memBytesProfileType, 8}}, work.values)
}

func TestWorkSampleWeight(t *testing.T) {
	kernel, _ := eventWork(event{kind: kindKernel, sampleWeight: 3, durationNS: 20})
	assert.Equal(t, []profileValue{{kernelTimeProfileType, 60}}, kernel.scaled().values)

	copyWork, _ := eventWork(event{kind: kindCopy, sampleWeight: 2, durationNS: 50,
		bytes: 4096})
	assert.Equal(t, []profileValue{{memTimeProfileType, 100}, {memBytesProfileType, 8192}},
		copyWork.scaled().values)

	// A weight of 0 counts as 1.
	unweighted, _ := eventWork(event{kind: kindKernel, durationNS: 20})
	assert.Equal(t, []profileValue{{kernelTimeProfileType, 20}}, unweighted.scaled().values)
}
