// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package gpu

import (
	"testing"

	cebpf "github.com/cilium/ebpf"
	"github.com/stretchr/testify/assert"

	"go.opentelemetry.io/ebpf-profiler/process"
)

func TestAPILibrary(t *testing.T) {
	loaded := &probe{apiStarts: &cebpf.Map{}}
	tests := map[string]string{
		"/usr/lib/libcudnn.so.9":         "libcudnn.so",
		"/opt/nvidia/libcublas.so.12":    "libcublas.so",
		"/opt/nvidia/libcublasLt.so.12":  "libcublasLt.so",
		"/usr/lib/libcudnn_graph.so.9":   "",
		"/usr/lib/x86_64-linux-gnu/libc": "",
	}
	for path, want := range tests {
		assert.Equal(t, want, loaded.apiLibrary(&process.RawMapping{Path: path}), path)
	}

	disabled := &probe{}
	assert.Empty(t, disabled.apiLibrary(&process.RawMapping{Path: "/usr/lib/libcudnn.so.9"}))
}

func TestAPIFunctionsAreUniquePerLibrary(t *testing.T) {
	seen := make(map[apiFunction]bool)
	for _, function := range apiFunctions {
		key := apiFunction{library: function.library, symbol: function.symbol}
		assert.False(t, seen[key], function.symbol)
		seen[key] = true
	}
}
