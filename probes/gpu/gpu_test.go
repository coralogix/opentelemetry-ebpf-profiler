// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package gpu

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"go.opentelemetry.io/ebpf-profiler/libpf"
)

func TestPreHandleTraceReportsWeight(t *testing.T) {
	p := &probe{}
	trace := &libpf.EbpfTrace{PID: 7, Value: 3<<32 | 41}
	assert.True(t, p.PreHandleTrace(trace))
	assert.Equal(t, int64(3), trace.Value, "the sample weight is reported")

	// A producer that does not sample leaves the weight at 0.
	trace = &libpf.EbpfTrace{PID: 7, Value: 42}
	p.PreHandleTrace(trace)
	assert.Equal(t, int64(1), trace.Value)
}
