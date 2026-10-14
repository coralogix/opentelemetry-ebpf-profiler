// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package gpu

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestKernelWork(t *testing.T) {
	work := kernelWork(kernelRecord{pid: 1, start: 10, end: 30, correlationID: 2}, "k", "")
	assert.Equal(t, gpuWork{
		pid:            1,
		correlationID:  2,
		leaf:           "k",
		values:         []profileValue{{kernelTimeProfileType, 20}},
		consumesLaunch: true,
	}, work)

	graph := kernelWork(kernelRecord{pid: 1, start: 10, end: 30, graphID: 4}, "", "")
	assert.False(t, graph.consumesLaunch)
	assert.Equal(t, unknownKernel, graph.leaf)

	assert.Empty(t, kernelWork(kernelRecord{start: 30, end: 30}, "k", "").values)
}

func TestWorkSampleWeight(t *testing.T) {
	kernel := kernelWork(kernelRecord{start: 10, end: 30, sampleWeight: 3}, "k", "")
	assert.Equal(t, []profileValue{{kernelTimeProfileType, 60}}, kernel.values)

	memcpy := memcpyWork(memcpyRecord{end: 50, bytes: 4096, sampleWeight: 2}, "")
	assert.Equal(t, []profileValue{{memTimeProfileType, 100}, {memBytesProfileType, 8192}},
		memcpy.values)

	alloc, ok := eventWork(eventRecord{kind: eventAlloc, v1: 64, sampleWeight: 4})
	assert.True(t, ok)
	assert.Equal(t, []profileValue{{allocBytesProfileType, 256}}, alloc.values)
}

func TestMemcpyWork(t *testing.T) {
	work := memcpyWork(memcpyRecord{pid: 1, start: 0, end: 50, copyKind: 1, bytes: 4096}, "")
	assert.Equal(t, "memcpy:HtoD", work.leaf)
	assert.False(t, work.consumesLaunch)
	assert.Equal(t, []profileValue{{memTimeProfileType, 50}, {memBytesProfileType, 4096}},
		work.values)

	assert.Equal(t, "memcpy:unknown", memcpyWork(memcpyRecord{copyKind: 99}, "").leaf)
}

func TestEventWork(t *testing.T) {
	alloc, ok := eventWork(eventRecord{pid: 1, kind: eventAlloc, v1: 1 << 20, v2: 3})
	assert.True(t, ok)
	assert.Equal(t, "alloc:device", alloc.leaf)
	assert.Equal(t, []profileValue{{allocBytesProfileType, 1 << 20}}, alloc.values)

	free, ok := eventWork(eventRecord{kind: eventFree, v1: 64, v2: 2})
	assert.True(t, ok)
	assert.Equal(t, "free:pinned", free.leaf)

	memset, ok := eventWork(eventRecord{kind: eventMemset, start: 5, end: 9, v1: 128})
	assert.True(t, ok)
	assert.Equal(t, "memset", memset.leaf)
	assert.Equal(t, []profileValue{{memTimeProfileType, 4}, {memBytesProfileType, 128}},
		memset.values)

	_, ok = eventWork(eventRecord{kind: 99})
	assert.False(t, ok)
}
