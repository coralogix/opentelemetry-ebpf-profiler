// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package tracer // import "go.opentelemetry.io/ebpf-profiler/tracer"

import (
	cebpf "github.com/cilium/ebpf"
	"github.com/cilium/ebpf/link"

	"go.opentelemetry.io/ebpf-profiler/libpf"
	"go.opentelemetry.io/ebpf-profiler/reporter/samples"
)

// TracerMaps is the tracer's loaded eBPF map set. Probes get a read-only
// view: never close or replace entries.
type TracerMaps = map[string]*cebpf.Map

// ReporterMetadata aliases the probe-origin metadata type so probe
// implementations only import this package.
type ReporterMetadata = samples.ProbeOriginMetadata

// Probe is a self-contained custom profiling source (upstream PR #1326
// pattern): it loads its own BPF programs and receives a dynamically
// assigned origin ID.
type Probe interface {
	// Load attaches the probe; the returned link's Close() tears it down.
	Load(origin libpf.Origin, maps TracerMaps) (link.Link, error)

	// ReportMetadata returns the probe's pprof sample-type metadata.
	ReportMetadata() ReporterMetadata
}
