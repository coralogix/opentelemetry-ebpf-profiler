// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package gpu

import (
	"runtime"
	"testing"

	"github.com/stretchr/testify/assert"

	"go.opentelemetry.io/ebpf-profiler/usdt"
)

func TestCheckArguments(t *testing.T) {
	pinned := map[string]string{"amd64": "8@%rax", "arm64": "8@x0"}[runtime.GOARCH]
	tests := map[string]struct {
		arguments string
		valid     bool
	}{
		"pinned register": {arguments: pinned, valid: true},
		"other register":  {arguments: "8@%rdi"},
		"arm64 other reg": {arguments: "8@x1"},
		"memory operand":  {arguments: "8@64(%rcx)"},
		"narrow register": {arguments: "4@%eax"},
		"two arguments":   {arguments: pinned + " " + pinned},
		"no arguments":    {arguments: ""},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			err := checkArguments(usdt.AttachmentPoint{
				Provider:  usdtProvider,
				Name:      "on_launch",
				Arguments: tc.arguments,
			})
			if tc.valid {
				assert.NoError(t, err)
			} else {
				assert.Error(t, err)
			}
		})
	}
}
