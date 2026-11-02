// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package gpu

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDemangleName(t *testing.T) {
	cache, err := newNameCache()
	require.NoError(t, err)
	for name, want := range map[string]string{
		// Parameter lists are dropped.
		"_Z10vector_addPfS_S_i": "vector_add",
		// Template lists are dropped, so instances share one name.
		"_Z9long_nameISt16integer_sequenceIiJLi0ELi1EEEEvPf":                                               "long_name",
		"_ZN2at6native29vectorized_elementwise_kernelILi4ENS0_11FillFunctorIfEESt5arrayIPcLm1EEEEviT0_T1_": "at::native::vectorized_elementwise_kernel",
		// Names that are not mangled, or no longer demangle, are kept.
		"ampere_sgemm_128x64_nn":                       "ampere_sgemm_128x64_nn",
		"_Z9long_nameISt16integer_sequenceIiJLi0ELi1E": "_Z9long_nameISt16integer_sequenceIiJLi0ELi1E",
	} {
		assert.Equal(t, want, cache.demangle(name), name)
		// The second lookup is served by the cache.
		assert.Equal(t, want, cache.demangle(name), name)
	}
}
