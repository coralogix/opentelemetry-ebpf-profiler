// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package libpf

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestFrameTypeFromString(t *testing.T) {
	// Simple check whether all FrameType values can be converted to string and back.
	for _, ft := range []FrameType{
		UnknownFrame, PHPFrame, PythonFrame, NativeFrame, KernelFrame, HotSpotFrame, RubyFrame,
		PerlFrame, V8Frame, DotnetFrame, LuaJITFrame, CUDAFrame, ROCmFrame} {
		t.Run(ft.String(), func(t *testing.T) {
			name := ft.String()
			result := FrameTypeFromString(name)
			require.Equal(t, ft, result)
		})
	}
}

func TestSynthesizedFrameTypes(t *testing.T) {
	// eBPF frame headers hold the frame type in 4 bits.
	const maxEbpfFrameType = 0xf
	for _, ft := range []FrameType{CUDAFrame, ROCmFrame} {
		require.Greater(t, ft, FrameType(maxEbpfFrameType))
		require.False(t, ft.IsError())
	}
	require.Equal(t, CUDA, CUDAFrame.Interpreter())
	require.Equal(t, "cuda", CUDAFrame.String())
	require.Equal(t, ROCm, ROCmFrame.Interpreter())
	require.Equal(t, "rocm", ROCmFrame.String())
}
