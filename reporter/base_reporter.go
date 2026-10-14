// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package reporter // import "go.opentelemetry.io/ebpf-profiler/reporter"

import (
	"errors"
	"fmt"
	"sync/atomic"
	"time"

	"go.opentelemetry.io/ebpf-profiler/internal/log"
	"go.opentelemetry.io/ebpf-profiler/libpf"
	"go.opentelemetry.io/ebpf-profiler/libpf/xsync"
	"go.opentelemetry.io/ebpf-profiler/reporter/internal/pdata"
	"go.opentelemetry.io/ebpf-profiler/reporter/samples"
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

	// snapshots holds the callbacks set by SetSnapshotSources, or nil.
	snapshots atomic.Pointer[snapshotHooks]
}

type snapshotHooks struct {
	collect     func() []samples.SnapshotProfile
	processMeta func(libpf.PID) samples.ProcessMeta
}

var errUnknownProfileType = errors.New("unknown trace profile type")

func (b *baseReporter) Stop() {
	b.runLoop.Stop()
}

func (b *baseReporter) ReportTraceEvent(trace *libpf.Trace, meta *samples.TraceEventMeta) error {
	if meta.ProfileType == nil {
		return fmt.Errorf("skip reporting trace: %w", errUnknownProfileType)
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
	traceHash := trace.Hash()

	eventsTree := b.traceEvents.WLock()
	defer b.traceEvents.WUnlock(&eventsTree)

	if _, exists := (*eventsTree)[key]; !exists {
		(*eventsTree)[key] = samples.ResourceToProfiles{
			EnvVars: meta.EnvVars,
			Events:  make(map[*samples.TypeMetadata]samples.SampleToEvents),
		}
	}

	rtp := (*eventsTree)[key]
	// Compared by hash to skip the map write-back when nothing changed.
	if meta.ResourceAttrs.Equivalent() != rtp.ResourceAttrs.Equivalent() {
		rtp.ResourceAttrs = meta.ResourceAttrs
		(*eventsTree)[key] = rtp
	}
	if _, exists := rtp.Events[meta.ProfileType]; !exists {
		rtp.Events[meta.ProfileType] = make(samples.SampleToEvents)
	}

	sampleKey := samples.SampleKey{
		Hash:      traceHash,
		Comm:      meta.Comm,
		TID:       int64(meta.TID),
		CPU:       int64(meta.CPU),
		SpanID:    meta.SpanID,
		TraceID:   meta.TraceID,
		ExtraMeta: extraMeta,
	}
	if events, exists := rtp.Events[meta.ProfileType][sampleKey]; exists {
		events.Timestamps = append(events.Timestamps, uint64(meta.Timestamp))
		events.Values = append(events.Values, meta.Value)
		return nil
	}

	rtp.Events[meta.ProfileType][sampleKey] = &samples.TraceEvents{
		Frames:     trace.Frames,
		Timestamps: []uint64{uint64(meta.Timestamp)},
		Values:     []int64{meta.Value},
		Labels:     trace.CustomLabels,
	}
	return nil
}

// SetSnapshotSources implements SnapshotMerger.
func (b *baseReporter) SetSnapshotSources(collect func() []samples.SnapshotProfile,
	processMeta func(libpf.PID) samples.ProcessMeta) {
	b.snapshots.Store(&snapshotHooks{collect: collect, processMeta: processMeta})
}

// mergeSnapshots folds probe-produced snapshot samples into tree so they are
// exported by the same path as trace events. Every snapshot sample is stamped
// with ts, the end of the collection interval.
func (b *baseReporter) mergeSnapshots(tree samples.TraceEventsTree, ts time.Time) {
	hooks := b.snapshots.Load()
	if hooks == nil {
		return
	}

	timestamp := uint64(ts.UnixNano())
	for _, profile := range hooks.collect() {
		if profile.ProfileType == nil {
			log.Warnf("Skipping snapshot profile without profile type")
			continue
		}
		for _, sample := range profile.Samples {
			var meta samples.ProcessMeta
			if sample.ProcessMeta != nil {
				meta = *sample.ProcessMeta
			} else {
				meta = hooks.processMeta(sample.PID)
			}
			events := snapshotEventsFor(tree, profile.ProfileType, sample.PID, meta)
			key := samples.SampleKey{Hash: sample.TraceHash}
			if existing, ok := events[key]; ok {
				existing.Timestamps = append(existing.Timestamps, timestamp)
				existing.Values = append(existing.Values, sample.Value)
				continue
			}
			events[key] = &samples.TraceEvents{
				Frames:     sample.Frames,
				Timestamps: []uint64{timestamp},
				Values:     []int64{sample.Value},
			}
		}
	}
}

// snapshotEventsFor returns the samples of profileType for the process's
// resource, creating the resource and profile type entries if needed.
func snapshotEventsFor(tree samples.TraceEventsTree, profileType *samples.TypeMetadata,
	pid libpf.PID, meta samples.ProcessMeta,
) samples.SampleToEvents {
	key := samples.ResourceKey{
		ContainerID:    meta.ContainerID,
		ExecutablePath: meta.ExecutablePath,
		PID:            int64(pid),
	}
	rtp, ok := tree[key]
	if !ok {
		rtp = samples.ResourceToProfiles{
			EnvVars:       meta.EnvVars,
			ResourceAttrs: meta.ResourceAttrs,
			Events:        make(map[*samples.TypeMetadata]samples.SampleToEvents),
		}
		tree[key] = rtp
	}

	events, ok := rtp.Events[profileType]
	if !ok {
		events = make(samples.SampleToEvents)
		rtp.Events[profileType] = events
	}
	return events
}
