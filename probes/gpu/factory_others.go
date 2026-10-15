// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//go:build !(linux && (amd64 || arm64))

package gpu // import "go.opentelemetry.io/ebpf-profiler/probes/gpu"

import (
	"context"
	"errors"

	"go.opentelemetry.io/collector/component"
	"go.opentelemetry.io/collector/extension"

	"go.opentelemetry.io/ebpf-profiler/probes/gpu/internal/metadata"
)

// NewFactory returns an extension.Factory for the gpu extension.
// The gpu extension is only functional on Linux amd64/arm64.
func NewFactory() extension.Factory {
	return extension.NewFactory(
		metadata.Type,
		func() component.Config { return &Config{} },
		func(_ context.Context, _ extension.Settings, _ component.Config) (extension.Extension, error) {
			return nil, errors.New("gpu extension is only supported on Linux amd64/arm64")
		},
		metadata.ExtensionStability,
	)
}
