// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//go:build linux && (amd64 || arm64)

package gpu // import "go.opentelemetry.io/ebpf-profiler/probes/gpu"

import (
	"context"

	"go.opentelemetry.io/collector/component"

	"go.opentelemetry.io/ebpf-profiler/tracer"
)

// gpuExtension is an OTel Collector extension that exposes a tracer.Probe
// for use by the ebpf_profiler receiver.
type gpuExtension struct {
	p tracer.Probe
}

func (e *gpuExtension) Start(_ context.Context, _ component.Host) error { return nil }
func (e *gpuExtension) Shutdown(_ context.Context) error                { return nil }

// Probe returns the tracer.Probe that this extension provides.
func (e *gpuExtension) Probe() tracer.Probe { return e.p }
