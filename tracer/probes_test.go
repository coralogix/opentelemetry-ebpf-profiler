// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package tracer

import (
	"testing"

	"github.com/stretchr/testify/assert"

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
