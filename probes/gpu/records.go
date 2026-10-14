// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package gpu // import "go.opentelemetry.io/ebpf-profiler/probes/gpu"

import (
	"bytes"
	"encoding/binary"

	"go.opentelemetry.io/ebpf-profiler/libpf"
)

// Record types and layouts sent by support/ebpf/gpu.ebpf.c. Both supported
// architectures are little-endian.
const (
	recordKernel uint32 = 1
	recordMemcpy uint32 = 2
	recordEvent  uint32 = 3
	recordError  uint32 = 4
)

const (
	// recordHeaderSize is the size of the {type, pid} header of every record.
	recordHeaderSize = 8
	// Kernel, memory copy and event records carry a 56-byte producer record.
	producerRecordSize = recordHeaderSize + 56
	errorMessageSize   = 120
	errorRecordSize    = recordHeaderSize + 8 + errorMessageSize
)

// Kinds of otel_gpu event records.
const (
	eventUVMHtoD     uint32 = 1
	eventUVMDtoH     uint32 = 2
	eventUVMCPUFault uint32 = 3
	eventUVMGPUFault uint32 = 4
	eventAlloc       uint32 = 5
	eventFree        uint32 = 6
	eventMemset      uint32 = 7
	eventStall       uint32 = 8
)

// kernelRecord is one kernel execution measured on the GPU.
type kernelRecord struct {
	pid           libpf.PID
	start, end    uint64
	correlationID uint32
	graphID       uint32
	namePtr       uint64
	rangesPtr     uint64
	sampleWeight  uint32
}

// memcpyRecord is one memory copy measured on the GPU.
type memcpyRecord struct {
	pid           libpf.PID
	start, end    uint64
	correlationID uint32
	copyKind      uint32
	bytes         uint64
	rangesPtr     uint64
	sampleWeight  uint32
}

// eventRecord is a lower-rate GPU event; kind selects the meaning of v1 and v2.
type eventRecord struct {
	pid           libpf.PID
	start, end    uint64
	kind          uint32
	correlationID uint32
	v1, v2        uint64
	namePtr       uint64
	sampleWeight  uint32
}

// errorRecord is a producer-side failure.
type errorRecord struct {
	pid     libpf.PID
	code    int32
	message string
}

// recordHeader returns the type and PID of a record.
func recordHeader(b []byte) (recordType uint32, pid libpf.PID, ok bool) {
	if len(b) < recordHeaderSize {
		return 0, 0, false
	}
	return binary.LittleEndian.Uint32(b[0:]), libpf.PID(binary.LittleEndian.Uint32(b[4:])), true
}

func decodeKernelRecord(pid libpf.PID, b []byte) (kernelRecord, bool) {
	if len(b) < producerRecordSize {
		return kernelRecord{}, false
	}
	le := binary.LittleEndian
	rec := b[recordHeaderSize:]
	return kernelRecord{
		pid:           pid,
		start:         le.Uint64(rec[0:]),
		end:           le.Uint64(rec[8:]),
		correlationID: le.Uint32(rec[16:]),
		graphID:       le.Uint32(rec[28:]),
		namePtr:       le.Uint64(rec[32:]),
		rangesPtr:     le.Uint64(rec[40:]),
		sampleWeight:  le.Uint32(rec[48:]),
	}, true
}

func decodeMemcpyRecord(pid libpf.PID, b []byte) (memcpyRecord, bool) {
	if len(b) < producerRecordSize {
		return memcpyRecord{}, false
	}
	le := binary.LittleEndian
	rec := b[recordHeaderSize:]
	return memcpyRecord{
		pid:           pid,
		start:         le.Uint64(rec[0:]),
		end:           le.Uint64(rec[8:]),
		correlationID: le.Uint32(rec[16:]),
		copyKind:      le.Uint32(rec[28:]),
		bytes:         le.Uint64(rec[32:]),
		rangesPtr:     le.Uint64(rec[40:]),
		sampleWeight:  le.Uint32(rec[48:]),
	}, true
}

func decodeEventRecord(pid libpf.PID, b []byte) (eventRecord, bool) {
	if len(b) < producerRecordSize {
		return eventRecord{}, false
	}
	le := binary.LittleEndian
	rec := b[recordHeaderSize:]
	return eventRecord{
		pid:           pid,
		start:         le.Uint64(rec[0:]),
		end:           le.Uint64(rec[8:]),
		kind:          le.Uint32(rec[16:]),
		correlationID: le.Uint32(rec[20:]),
		v1:            le.Uint64(rec[24:]),
		v2:            le.Uint64(rec[32:]),
		namePtr:       le.Uint64(rec[40:]),
		sampleWeight:  le.Uint32(rec[48:]),
	}, true
}

func decodeErrorRecord(pid libpf.PID, b []byte) (errorRecord, bool) {
	if len(b) < errorRecordSize {
		return errorRecord{}, false
	}
	message := b[recordHeaderSize+8 : errorRecordSize]
	if end := bytes.IndexByte(message, 0); end >= 0 {
		message = message[:end]
	}
	return errorRecord{
		pid:     pid,
		code:    int32(binary.LittleEndian.Uint32(b[recordHeaderSize:])),
		message: string(message),
	}, true
}
