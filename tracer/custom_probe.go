// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package tracer // import "go.opentelemetry.io/ebpf-profiler/tracer"

import (
	cebpf "github.com/cilium/ebpf"
	"github.com/cilium/ebpf/link"

	"go.opentelemetry.io/ebpf-profiler/libpf"
	"go.opentelemetry.io/ebpf-profiler/reporter/samples"
)

// TracerMaps is the set of loaded eBPF maps owned by the main tracer.
// Custom probes receive a read-only view; they must not close or replace maps.
type TracerMaps = map[string]*cebpf.Map

// ReporterMetadata is an alias for the probe-origin metadata type in the
// reporter/samples package. Defined here so probe implementations only need
// to import this package, not reporter/samples.
type ReporterMetadata = samples.ProbeOriginMetadata

// SystemVariables carries kernel offsets resolved at startup. Probes that
// hook scheduling events need these; GPU/uprobe probes typically ignore them.
type SystemVariables struct {
	TPBaseOffset      uint64
	TaskStackOffset   uint32
	StackPtregsOffset uint32
}

// Probe is implemented by any custom profiling source that integrates with
// the tracer. The pattern follows the RFC in PR #1326 of the upstream repo:
// each probe is self-contained, loads its own BPF programs, and receives a
// dynamically assigned origin ID so the reporter can emit correct sample types.
type Probe interface {
	// Load attaches the probe. The tracer assigns origin so the probe can tag
	// emitted samples. maps is the tracer's shared map collection (read-only).
	// sysVars carries kernel offsets needed by scheduling probes.
	// Returns a link whose Close() tears down all probe resources.
	Load(origin libpf.Origin, maps TracerMaps, sysVars *SystemVariables) (link.Link, error)

	// ReportMetadata returns pprof sample-type metadata for this probe.
	// Called once by Enable() to register the origin with the reporter.
	ReportMetadata() ReporterMetadata
}
