// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//go:build integration && linux && static_build

package gpu_test

import (
	_ "embed"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

// producerBinary is the fake producer, built by `make gpu-test-producer`: the
// VMs that run the static test binaries have no compiler.
//
//go:embed testdata/producer
var producerBinary []byte

// buildProducer writes out the fake producer.
func buildProducer(t *testing.T) string {
	t.Helper()
	bin := filepath.Join(t.TempDir(), "producer")
	require.NoError(t, os.WriteFile(bin, producerBinary, 0o755))
	return bin
}
