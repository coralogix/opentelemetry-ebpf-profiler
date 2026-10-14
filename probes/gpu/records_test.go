// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package gpu

import (
	"encoding/binary"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"go.opentelemetry.io/ebpf-profiler/libpf"
	"go.opentelemetry.io/ebpf-profiler/remotememory"
)

func TestDecodeKernelRecord(t *testing.T) {
	b := make([]byte, producerRecordSize)
	le := binary.LittleEndian
	le.PutUint32(b[0:], recordKernel)
	le.PutUint32(b[4:], 42)
	le.PutUint64(b[8:], 1000)
	le.PutUint64(b[16:], 1500)
	le.PutUint32(b[24:], 7)
	le.PutUint32(b[36:], 3)
	le.PutUint64(b[40:], 0xdead)
	le.PutUint64(b[48:], 0xbeef)
	le.PutUint32(b[56:], 5)

	recordType, pid, ok := recordHeader(b)
	require.True(t, ok)
	assert.Equal(t, recordKernel, recordType)
	assert.Equal(t, libpf.PID(42), pid)

	rec, ok := decodeKernelRecord(pid, b)
	require.True(t, ok)
	assert.Equal(t, kernelRecord{
		pid:           42,
		start:         1000,
		end:           1500,
		correlationID: 7,
		graphID:       3,
		namePtr:       0xdead,
		rangesPtr:     0xbeef,
		sampleWeight:  5,
	}, rec)

	_, ok = decodeKernelRecord(pid, b[:producerRecordSize-1])
	assert.False(t, ok)
	_, _, ok = recordHeader(b[:recordHeaderSize-1])
	assert.False(t, ok)
}

func TestDemangleName(t *testing.T) {
	tests := map[string]string{
		"_Z10vector_addPfS_S_i": "vector_add",
		"_ZN7cutlass7Kernel2INS_4gemm6kernel4GemmIiEEEEvNT_6ParamsE": "cutlass::Kernel2",
		"flash_attn_ext_f16": "flash_attn_ext_f16",
		"":                   "",
	}
	for mangled, want := range tests {
		assert.Equal(t, want, demangleName(mangled), mangled)
	}
}

// pagedMemory is process memory whose readable range ends at limit.
type pagedMemory struct {
	data  []byte
	limit int64
}

func (m pagedMemory) ReadAt(p []byte, off int64) (int, error) {
	if off+int64(len(p)) > m.limit {
		return 0, errors.New("unmapped")
	}
	return copy(p, m.data[off:]), nil
}

func TestReadStringAtMappingEnd(t *testing.T) {
	const name = "_Z10vector_addPfS_S_i"
	data := make([]byte, 2*4096)
	offset := 4096 - len(name) - 1
	copy(data[offset:], name)
	rm := remotememory.RemoteMemory{ReaderAt: pagedMemory{data: data, limit: 4096}}

	assert.Equal(t, name, readString(rm, uint64(offset)))

	// A string continuing into the next readable page.
	rm = remotememory.RemoteMemory{ReaderAt: pagedMemory{data: data, limit: int64(len(data))}}
	copy(data[4090:], "spans pages\x00")
	assert.Equal(t, "spans pages", readString(rm, 4090))

	// Unterminated within the readable range.
	for i := range data {
		data[i] = 'x'
	}
	assert.Empty(t, readString(rm, 0))
}

func TestDecodeMemcpyAndEventRecords(t *testing.T) {
	b := make([]byte, producerRecordSize)
	le := binary.LittleEndian
	le.PutUint64(b[8:], 100)
	le.PutUint64(b[16:], 160)
	le.PutUint32(b[24:], 9)
	le.PutUint32(b[36:], 2)
	le.PutUint64(b[40:], 4096)
	le.PutUint64(b[48:], 0xbeef)
	le.PutUint32(b[56:], 2)

	memcpy, ok := decodeMemcpyRecord(5, b)
	require.True(t, ok)
	assert.Equal(t, memcpyRecord{
		pid: 5, start: 100, end: 160, correlationID: 9, copyKind: 2, bytes: 4096, rangesPtr: 0xbeef,
		sampleWeight: 2,
	}, memcpy)

	le.PutUint32(b[24:], eventAlloc)
	le.PutUint32(b[28:], 11)
	le.PutUint64(b[32:], 1<<20)
	le.PutUint64(b[40:], 3)
	event, ok := decodeEventRecord(5, b)
	require.True(t, ok)
	assert.Equal(t, eventRecord{
		pid: 5, start: 100, end: 160, kind: eventAlloc, correlationID: 11,
		v1: 1 << 20, v2: 3, namePtr: 0xbeef, sampleWeight: 2,
	}, event)
}

func TestDecodeErrorRecord(t *testing.T) {
	b := make([]byte, errorRecordSize)
	binary.LittleEndian.PutUint32(b[recordHeaderSize:], 0xffffffff)
	copy(b[recordHeaderSize+8:], "cuptiSubscribe failed\x00garbage")
	rec, ok := decodeErrorRecord(7, b)
	require.True(t, ok)
	assert.Equal(t, errorRecord{pid: 7, code: -1, message: "cuptiSubscribe failed"}, rec)

	_, ok = decodeErrorRecord(7, b[:errorRecordSize-1])
	assert.False(t, ok)
}
