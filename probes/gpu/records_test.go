// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package gpu

import (
	"encoding/binary"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/encoding/protowire"

	"go.opentelemetry.io/ebpf-profiler/libpf"
)

// encodeEvent lays out an event as support/ebpf/gpu.ebpf.c sends it.
func encodeEvent(pid uint32, correlationID uint64, sampleWeight uint32, kind uint16,
	payload []byte,
) []byte {
	b := make([]byte, eventHeaderSize, eventHeaderSize+len(payload))
	le := binary.LittleEndian
	le.PutUint32(b[0:], pid)
	le.PutUint64(b[8:], correlationID)
	le.PutUint32(b[16:], sampleWeight)
	le.PutUint16(b[20:], kind)
	le.PutUint16(b[22:], uint16(len(payload)))
	return append(b, payload...)
}

func appendString(b []byte, num protowire.Number, s string) []byte {
	b = protowire.AppendTag(b, num, protowire.BytesType)
	return protowire.AppendString(b, s)
}

func appendUint(b []byte, num protowire.Number, v uint64) []byte {
	b = protowire.AppendTag(b, num, protowire.VarintType)
	return protowire.AppendVarint(b, v)
}

func TestDecodeEvent(t *testing.T) {
	var payload []byte
	payload = appendString(payload, fieldGPU, "GPU-1")
	payload = appendString(payload, fieldName, "HtoD")
	payload = appendString(payload, fieldRanges, "aten::copy_")
	payload = appendString(payload, fieldRanges, "train_step")
	payload = appendUint(payload, fieldDurationNS, 1500)
	payload = appendUint(payload, fieldBytes, 4096)

	ev, err := decodeEvent(encodeEvent(7, 1<<40|42, 5, kindCopy, payload))
	require.NoError(t, err)
	assert.Equal(t, event{
		pid:           libpf.PID(7),
		correlationID: 1<<40 | 42,
		sampleWeight:  5,
		kind:          kindCopy,
		gpu:           "GPU-1",
		name:          "HtoD",
		ranges:        "aten::copy_" + rangeSeparator + "train_step",
		durationNS:    1500,
		bytes:         4096,
	}, ev)
}

func TestDecodeEventSkipsUnknownFields(t *testing.T) {
	var payload []byte
	payload = appendString(payload, 99, "a later field")
	payload = appendUint(payload, 100, 7)
	payload = protowire.AppendTag(payload, 101, protowire.Fixed64Type)
	payload = protowire.AppendFixed64(payload, 1)
	payload = appendString(payload, fieldName, "k")
	// A known field number with another wire type is skipped too.
	payload = appendUint(payload, fieldName, 1)

	ev, err := decodeEvent(encodeEvent(1, 2, 0, kindKernel, payload))
	require.NoError(t, err)
	assert.Equal(t, "k", ev.name)
}

func TestDecodeEventErrors(t *testing.T) {
	full := encodeEvent(1, 2, 0, kindKernel, appendString(nil, fieldName, "kernel"))

	_, err := decodeEvent(full[:eventHeaderSize-1])
	require.ErrorIs(t, err, errShortEvent, "shorter than the header")

	_, err = decodeEvent(full[:len(full)-1])
	require.ErrorIs(t, err, errShortEvent, "shorter than its payload size")

	truncated := encodeEvent(1, 2, 0, kindKernel, appendString(nil, fieldName, "kernel")[:4])
	_, err = decodeEvent(truncated)
	require.Error(t, err, "a string cut inside the payload")

	ev, err := decodeEvent(encodeEvent(1, 2, 3, kindKernel, nil))
	require.NoError(t, err, "every payload field is optional")
	assert.Equal(t, uint32(3), ev.sampleWeight)
}
