// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

//go:build integration && linux && !static_build

package gpu_test

import (
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

// buildProducer builds the fake producer.
func buildProducer(t *testing.T) string {
	t.Helper()
	bin := filepath.Join(t.TempDir(), "producer")
	out, err := exec.Command("cc", "-O2", "-g", "-o", bin, "testdata/producer.c").CombinedOutput()
	require.NoError(t, err, string(out))
	return bin
}
