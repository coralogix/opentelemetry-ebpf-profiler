// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package reporter // import "go.opentelemetry.io/ebpf-profiler/reporter"

import (
	"errors"
	"fmt"
	"sync"
	"time"

	"go.opentelemetry.io/ebpf-profiler/libpf"
	"go.opentelemetry.io/ebpf-profiler/libpf/xsync"
	"go.opentelemetry.io/ebpf-profiler/reporter/internal/pdata"
	"go.opentelemetry.io/ebpf-profiler/reporter/samples"
	"go.opentelemetry.io/ebpf-profiler/support"
)

// baseReporter encapsulates shared behavior between all the available reporters.
type baseReporter struct {
	cfg *Config

	// name is the ScopeProfile's name.
	name string

	// version is the ScopeProfile's version.
	version string

	// runLoop handles the run loop
	runLoop *runLoop

	// pdata holds the generator for the data being exported.
	pdata *pdata.Pdata

	// traceEvents stores reported trace events (trace metadata with frames and counts)
	traceEvents xsync.RWMutex[samples.TraceEventsTree]

	// collectionStartTime tracks when the current collection window started.
	// Initialized when Start() is called. The duration of the first profile may be
	// slightly overestimated as it includes tracer setup time before samples arrive.
	collectionStartTime time.Time

	// probeOriginsMu guards probeOrigins and probeOriginsVersion.
	probeOriginsMu sync.RWMutex
	// probeOrigins holds metadata for dynamically registered probe origins
	// (registered via RegisterProbeOrigin).
	probeOrigins map[libpf.Origin]samples.ProbeOriginMetadata
	// probeOriginsVersion is incremented on every RegisterProbeOrigin call.
	// syncProbeOriginsToPdata uses it to skip the copy when nothing changed.
	probeOriginsVersion uint64
	// probeOriginsLastVer is the version last synced to pdata.ProbeOrigins.
	probeOriginsLastVer uint64
}

var errUnknownOrigin = errors.New("unknown trace origin")

func (b *baseReporter) Stop() {
	b.runLoop.Stop()
}

// RegisterProbeOrigin implements reporter.ProbeRegistrar.
func (b *baseReporter) RegisterProbeOrigin(origin libpf.Origin, meta samples.ProbeOriginMetadata) error {
	b.probeOriginsMu.Lock()
	defer b.probeOriginsMu.Unlock()
	if b.probeOrigins == nil {
		b.probeOrigins = make(map[libpf.Origin]samples.ProbeOriginMetadata)
	}
	b.probeOrigins[origin] = meta
	b.probeOriginsVersion++
	return nil
}

// syncProbeOriginsToPdata copies the current probe origin map into p.ProbeOrigins
// so that the next Generate() call emits correct sample types for custom probes.
// Call this immediately before Generate(). The copy is skipped when the set of
// registered origins has not changed since the last sync (common case after startup).
func (b *baseReporter) syncProbeOriginsToPdata() {
	b.probeOriginsMu.RLock()
	ver := b.probeOriginsVersion
	b.probeOriginsMu.RUnlock()

	if ver == b.probeOriginsLastVer {
		return
	}

	b.probeOriginsMu.RLock()
	defer b.probeOriginsMu.RUnlock()
	out := make(map[libpf.Origin]samples.ProbeOriginMetadata, len(b.probeOrigins))
	for k, v := range b.probeOrigins {
		out[k] = v
	}
	b.pdata.ProbeOrigins = out
	b.probeOriginsLastVer = ver
}

func (b *baseReporter) ReportTraceEvent(trace *libpf.Trace, meta *samples.TraceEventMeta) error {
	switch meta.Origin {
	case support.TraceOriginSampling:
	case support.TraceOriginOffCPU:
	case support.TraceOriginProbe:
	case support.TraceOriginGPU:
	default:
		// Accept dynamically registered probe origins.
		b.probeOriginsMu.RLock()
		_, known := b.probeOrigins[meta.Origin]
		b.probeOriginsMu.RUnlock()
		if !known {
			return fmt.Errorf("skip reporting trace for %d origin: %w", meta.Origin,
				errUnknownOrigin)
		}
	}

	var extraMeta any
	if b.cfg.ExtraSampleAttrProd != nil {
		extraMeta = b.cfg.ExtraSampleAttrProd.CollectExtraSampleMeta(trace, meta)
	}

	key := samples.ResourceKey{
		APMServiceName: meta.APMServiceName,
		ContainerID:    meta.ContainerID,
		PID:            int64(meta.PID),
		ExecutablePath: meta.ExecutablePath,
	}

	eventsTree := b.traceEvents.WLock()
	defer b.traceEvents.WUnlock(&eventsTree)

	if _, exists := (*eventsTree)[key]; !exists {
		(*eventsTree)[key] = samples.ResourceToProfiles{
			EnvVars: meta.EnvVars,
			Events:  make(map[libpf.Origin]samples.SampleToEvents),
		}
	}

	rtp := (*eventsTree)[key]
	if _, exists := rtp.Events[meta.Origin]; !exists {
		rtp.Events[meta.Origin] = make(samples.SampleToEvents)
	}

	sampleKey := samples.SampleKey{
		Hash:      trace.Hash,
		Comm:      meta.Comm,
		TID:       int64(meta.TID),
		CPU:       int64(meta.CPU),
		SpanID:    meta.SpanID,
		TraceID:   meta.TraceID,
		ExtraMeta: extraMeta,
	}
	if events, exists := rtp.Events[meta.Origin][sampleKey]; exists {
		events.Timestamps = append(events.Timestamps, uint64(meta.Timestamp))
		events.Values = append(events.Values, meta.Value)
		return nil
	}

	rtp.Events[meta.Origin][sampleKey] = &samples.TraceEvents{
		Frames:     trace.Frames,
		Timestamps: []uint64{uint64(meta.Timestamp)},
		Values:     []int64{meta.Value},
		Labels:     trace.CustomLabels,
	}
	return nil
}
