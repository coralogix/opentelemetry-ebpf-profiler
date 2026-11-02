// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package gpu // import "go.opentelemetry.io/ebpf-profiler/probes/gpu"

import (
	"encoding/binary"
	"errors"
	"strings"

	"google.golang.org/protobuf/encoding/protowire"

	"go.opentelemetry.io/ebpf-profiler/libpf"
)

// Kinds of events, from enum otel_gpu_kind in otel_gpu_v1_usdt.h. LAUNCH events
// never get here: support/ebpf/gpu.ebpf.c turns them into traces.
const (
	kindKernel uint16 = 2
	kindCopy   uint16 = 3
	kindFill   uint16 = 4
)

// Fields of the payload, from enum otel_gpu_field in otel_gpu_v1_usdt.h.
const (
	fieldGPU        protowire.Number = 1
	fieldName       protowire.Number = 2
	fieldRanges     protowire.Number = 3
	fieldDurationNS protowire.Number = 4
	fieldBytes      protowire.Number = 5
)

// eventHeaderSize is what support/ebpf/gpu.ebpf.c sends before the payload: the
// PID, 4 bytes of padding, and the producer's 16-byte event header.
const eventHeaderSize = 24

// event is an event of a producer, as the gpu probe receives it.
type event struct {
	pid           libpf.PID
	correlationID uint64
	sampleWeight  uint32
	kind          uint16
	gpu           string
	name          string
	// ranges holds the names of the work in progress, innermost first, joined
	// by rangeSeparator.
	ranges     string
	durationNS uint64
	bytes      uint64
}

var errShortEvent = errors.New("event shorter than its header and payload")

// decodeEvent decodes an event sent by eBPF. Both supported architectures are
// little endian.
func decodeEvent(b []byte) (event, error) {
	if len(b) < eventHeaderSize {
		return event{}, errShortEvent
	}
	le := binary.LittleEndian
	ev := event{
		pid:           libpf.PID(le.Uint32(b[0:])),
		correlationID: le.Uint64(b[8:]),
		sampleWeight:  le.Uint32(b[16:]),
		kind:          le.Uint16(b[20:]),
	}
	end := eventHeaderSize + int(le.Uint16(b[22:]))
	if len(b) < end {
		return event{}, errShortEvent
	}
	if err := ev.decodePayload(b[eventHeaderSize:end]); err != nil {
		return event{}, err
	}
	return ev, nil
}

// decodePayload reads the fields of a Payload message, skipping the ones it
// does not know.
func (ev *event) decodePayload(b []byte) error {
	var ranges []string
	for len(b) > 0 {
		num, typ, n := protowire.ConsumeTag(b)
		if n < 0 {
			return protowire.ParseError(n)
		}
		b = b[n:]

		switch {
		case typ == protowire.BytesType && isStringField(num):
			value, n := protowire.ConsumeBytes(b)
			if n < 0 {
				return protowire.ParseError(n)
			}
			b = b[n:]
			switch num {
			case fieldGPU:
				ev.gpu = string(value)
			case fieldName:
				ev.name = string(value)
			default:
				ranges = append(ranges, string(value))
			}
		case typ == protowire.VarintType && (num == fieldDurationNS || num == fieldBytes):
			value, n := protowire.ConsumeVarint(b)
			if n < 0 {
				return protowire.ParseError(n)
			}
			b = b[n:]
			if num == fieldDurationNS {
				ev.durationNS = value
			} else {
				ev.bytes = value
			}
		default:
			n := protowire.ConsumeFieldValue(num, typ, b)
			if n < 0 {
				return protowire.ParseError(n)
			}
			b = b[n:]
		}
	}
	ev.ranges = strings.Join(ranges, rangeSeparator)
	return nil
}

func isStringField(num protowire.Number) bool {
	return num == fieldGPU || num == fieldName || num == fieldRanges
}
