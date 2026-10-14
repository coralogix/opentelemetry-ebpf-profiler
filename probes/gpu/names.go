// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package gpu // import "go.opentelemetry.io/ebpf-profiler/probes/gpu"

import (
	"bytes"
	"strings"
	"sync"

	"github.com/ianlancetaylor/demangle"

	"go.opentelemetry.io/ebpf-profiler/libpf"
	"go.opentelemetry.io/ebpf-profiler/remotememory"
)

// nameCache resolves strings that producers pass as pointers into their own
// memory. The contract keeps them valid for the process lifetime, so results
// are cached by PID and pointer until the process exits.
type nameCache struct {
	mu    sync.RWMutex
	names map[nameKey]string
}

type nameKey struct {
	pid libpf.PID
	ptr uint64
}

func newNameCache() *nameCache {
	return &nameCache{names: make(map[nameKey]string)}
}

// resolve returns the demangled string at ptr in the process, or "" if it
// cannot be read.
func (c *nameCache) resolve(pid libpf.PID, ptr uint64) string {
	if ptr == 0 {
		return ""
	}
	key := nameKey{pid: pid, ptr: ptr}
	c.mu.RLock()
	name, ok := c.names[key]
	c.mu.RUnlock()
	if ok {
		return name
	}

	name = demangleName(readString(remotememory.NewProcessVirtualMemory(pid), ptr))
	c.mu.Lock()
	c.names[key] = name
	c.mu.Unlock()
	return name
}

// readString reads the NUL-terminated string at addr, up to maxNameLength bytes.
// It reads page by page: one read crossing into an unmapped page fails whole,
// which RemoteMemory.String does not handle for strings near a mapping's end.
func readString(rm remotememory.RemoteMemory, addr uint64) string {
	const (
		pageSize      = 4096
		maxNameLength = 4096
	)
	var name []byte
	var chunk [pageSize]byte
	for len(name) < maxNameLength {
		n := pageSize - addr%pageSize
		if _, err := rm.ReadAt(chunk[:n], int64(addr)); err != nil {
			return ""
		}
		if end := bytes.IndexByte(chunk[:n], 0); end >= 0 {
			return string(append(name, chunk[:end]...))
		}
		name = append(name, chunk[:n]...)
		addr += n
	}
	return ""
}

// forget drops the names of an exited process.
func (c *nameCache) forget(pid libpf.PID) {
	c.mu.Lock()
	defer c.mu.Unlock()
	for key := range c.names {
		if key.pid == pid {
			delete(c.names, key)
		}
	}
}

// demangleName demangles an Itanium C++ symbol without parameter and template
// lists, which would otherwise split one kernel into many names.
func demangleName(name string) string {
	if !strings.HasPrefix(name, "_Z") {
		return name
	}
	return demangle.Filter(name, demangle.NoParams, demangle.NoTemplateParams)
}
