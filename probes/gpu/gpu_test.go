// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package gpu

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"go.opentelemetry.io/ebpf-profiler/libpf"
	"go.opentelemetry.io/ebpf-profiler/reporter/samples"
)

func TestPreHandleTraceKeepsCorrelationID(t *testing.T) {
	p := &probe{}
	trace := &libpf.EbpfTrace{PID: 7, ContextValues: []uint64{1<<40 | 41, 3}}
	assert.True(t, p.PreHandleTrace(trace))
	assert.Equal(t, libpf.PID(7), p.launch.pid)
	assert.Equal(t, uint32(41), p.launch.correlationID)

	assert.False(t, p.PreHandleTrace(&libpf.EbpfTrace{PID: 7}), "a launch without its values")
}

func TestLaunchWeight(t *testing.T) {
	weight := func(values ...uint64) []int64 {
		return appendLaunchWeight(nil, &samples.TraceEventMeta{ContextValues: values})
	}
	assert.Equal(t, []int64{3}, weight(41, 3))
	// A producer that does not sample leaves the weight at 0.
	assert.Equal(t, []int64{1}, weight(42, 0))
}
