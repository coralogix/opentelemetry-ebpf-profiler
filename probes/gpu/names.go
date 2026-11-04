// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package gpu // import "go.opentelemetry.io/ebpf-profiler/probes/gpu"

import (
	"strings"

	lru "github.com/elastic/go-freelru"
	"github.com/ianlancetaylor/demangle"
	"github.com/zeebo/xxh3"
)

// nameCapacity bounds the kernel names whose demangled form is kept.
const nameCapacity = 1 << 12

// nameCache keeps the demangled form of kernel names, which repeat at every
// launch. It is used only by the goroutine reading events.
type nameCache struct {
	names *lru.LRU[string, string]
}

func newNameCache() (*nameCache, error) {
	names, err := lru.New[string, string](nameCapacity, func(s string) uint32 {
		return uint32(xxh3.HashString(s))
	})
	if err != nil {
		return nil, err
	}
	return &nameCache{names: names}, nil
}

// demangle returns the kernel name without parameter and template lists.
func (c *nameCache) demangle(name string) string {
	if demangled, ok := c.names.Get(name); ok {
		return demangled
	}
	demangled := demangleName(name)
	c.names.Add(name, demangled)
	return demangled
}

// demangleName demangles an Itanium C++ symbol without parameter and template
// lists, which would otherwise split one kernel into many names.
func demangleName(name string) string {
	if !strings.HasPrefix(name, "_Z") {
		return name
	}
	return demangle.Filter(name, demangle.NoParams, demangle.NoTemplateParams)
}
