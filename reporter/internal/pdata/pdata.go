// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package pdata // import "go.opentelemetry.io/ebpf-profiler/reporter/internal/pdata"

import (
	"go.opentelemetry.io/ebpf-profiler/libpf"
	"go.opentelemetry.io/ebpf-profiler/reporter/samples"
)

// Pdata holds the cache for the data used to generate the events reporters
// will export when handling OTLP data.
type Pdata struct {
	// ExtraSampleAttrProd is an optional hook point for adding custom
	// attributes to samples.
	ExtraSampleAttrProd samples.SampleAttrProducer

	// samplesPerSecond is the number of samples per second.
	samplesPerSecond int

	// ProbeOrigins holds metadata for dynamically registered probe origins.
	// Set by the base reporter before each Generate() call via SetProbeOrigins.
	ProbeOrigins map[libpf.Origin]samples.ProbeOriginMetadata
}

func New(samplesPerSecond int, extra samples.SampleAttrProducer) (*Pdata, error) {
	return &Pdata{
		samplesPerSecond:    samplesPerSecond,
		ExtraSampleAttrProd: extra,
	}, nil
}

// Purge purges all the expired data
func (p *Pdata) Purge() {
}
