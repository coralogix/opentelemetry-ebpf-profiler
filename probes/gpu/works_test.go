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

	// Unified memory counters have no call site to sample.
	uvm, ok := eventWork(eventRecord{kind: eventUVMHtoD, v1: 8, sampleWeight: 4})
	assert.True(t, ok)
	assert.Equal(t, []profileValue{{uvmBytesProfileType, 8}}, uvm.values)
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

func TestUnifiedMemoryWork(t *testing.T) {
	tests := map[uint32]struct {
		leaf        string
		profileType string
	}{
		eventUVMHtoD:     {"uvm:HtoD", "gpu_uvm_bytes"},
		eventUVMDtoH:     {"uvm:DtoH", "gpu_uvm_bytes"},
		eventUVMCPUFault: {"uvm:cpu_fault", "gpu_uvm_faults"},
		eventUVMGPUFault: {"uvm:gpu_fault", "gpu_uvm_faults"},
	}
	for kind, want := range tests {
		work, ok := eventWork(eventRecord{pid: 1, kind: kind, v1: 8})
		assert.True(t, ok)
		assert.True(t, work.noLaunch)
		assert.Equal(t, want.leaf, work.leaf)
		assert.Len(t, work.values, 1)
		assert.Equal(t, want.profileType, work.values[0].profileType.SampleType)
		assert.Equal(t, int64(8), work.values[0].value)
	}
}

func TestStallWork(t *testing.T) {
	work := stallWork(eventRecord{pid: 1, kind: eventStall, v2: 12}, "vector_add",
		"memory_dependency")
	assert.Equal(t, gpuWork{
		pid:      1,
		leaf:     "stall:memory_dependency",
		function: "vector_add",
		noLaunch: true,
		values:   []profileValue{{stallsProfileType, 12}},
	}, work)

	assert.Empty(t, stallWork(eventRecord{v2: 12}, "", "reason").values)
	assert.Empty(t, stallWork(eventRecord{v2: 12}, "fn", "").values)
}

func TestCollectiveWork(t *testing.T) {
	work := collectiveWork(eventRecord{pid: 1, kind: eventCollective, start: 100, end: 400, v1: 1 << 20},
		"AllReduce")
	assert.Equal(t, gpuWork{
		pid:      1,
		leaf:     "collective:AllReduce",
		noLaunch: true,
		values: []profileValue{
			{collectiveTimeProfileType, 300}, {collectiveBytesProfileType, 1 << 20},
		},
	}, work)
	assert.Equal(t, "collective:unknown", collectiveWork(eventRecord{}, "").leaf)
}

func TestAPIWork(t *testing.T) {
	work := apiWork(apiSpanRecord{pid: 1, start: 100, end: 350, cookie: 0})
	assert.Equal(t, gpuWork{
		pid:      1,
		leaf:     "cudnn:ConvolutionForward",
		noLaunch: true,
		values:   []profileValue{{apiTimeProfileType, 250}},
	}, work)
	assert.Equal(t, "api:unknown", apiWork(apiSpanRecord{end: 1, cookie: 999}).leaf)

	stale := apiSpanRecord{start: 1, end: 2 + uint64(maxAPISpan)}
	assert.Empty(t, apiWork(stale).values)
}
