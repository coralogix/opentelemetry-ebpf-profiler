// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package tracer

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"go.opentelemetry.io/ebpf-profiler/libpf"
	"go.opentelemetry.io/ebpf-profiler/reporter/samples"
)

type staticSnapshotSource []samples.SnapshotProfile

func (s staticSnapshotSource) ProduceSnapshots() []samples.SnapshotProfile { return s }

func TestCollectSnapshots(t *testing.T) {
	first := samples.SnapshotProfile{ProfileType: &samples.TypeMetadata{SampleType: "a"}}
	second := samples.SnapshotProfile{ProfileType: &samples.TypeMetadata{SampleType: "b"}}
	trc := &Tracer{snapshotSources: []SnapshotSource{
		staticSnapshotSource{first},
		staticSnapshotSource{},
		staticSnapshotSource{second},
	}}

	assert.Equal(t, []samples.SnapshotProfile{first, second}, trc.CollectSnapshots())
	assert.Empty(t, (&Tracer{}).CollectSnapshots())
}

type handlerProbe struct {
	Probe
	origins []uint16
}

func (h *handlerProbe) PreOrigins() []uint16                 { return h.origins }
func (h *handlerProbe) PreHandleTrace(*libpf.EbpfTrace) bool { return true }
func (h *handlerProbe) PostOrigins() []uint16                { return h.origins }
func (h *handlerProbe) PostHandleTrace(*libpf.Trace)         {}

func TestAddTraceHandlersPublishesCopies(t *testing.T) {
	trc := &Tracer{}
	trc.traceHandlers.Store(&traceHandlers{
		pre:  make(map[uint16][]PreTraceHandler),
		post: make(map[uint16][]PostTraceHandler),
	})
	first := &handlerProbe{origins: []uint16{1}}
	trc.addTraceHandlers(first)
	published := trc.traceHandlers.Load()

	second := &handlerProbe{origins: []uint16{1, 2}}
	trc.addTraceHandlers(second)

	// What HandleTrace may still be reading is left untouched.
	assert.Equal(t, []PreTraceHandler{first}, published.pre[1])
	assert.Empty(t, published.pre[2])

	current := trc.traceHandlers.Load()
	assert.Equal(t, []PreTraceHandler{first, second}, current.pre[1])
	assert.Equal(t, []PostTraceHandler{first, second}, current.post[1])
	assert.Equal(t, []PreTraceHandler{second}, current.pre[2])
}
