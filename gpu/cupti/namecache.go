// SPDX-License-Identifier: Apache-2.0

package cupti // import "go.opentelemetry.io/ebpf-profiler/gpu/cupti"

import (
	"strings"
	"sync"

	"github.com/ianlancetaylor/demangle"

	"go.opentelemetry.io/ebpf-profiler/libpf"
	"go.opentelemetry.io/ebpf-profiler/remotememory"
)

// NameCache resolves and memoises strings the shim passes as pointers into
// the workload's address space. CUPTI keeps them alive for the process
// lifetime, so caching by (pid, ptr) is safe — as long as entries die with
// the PID (ForgetPID/Sweep), or a reused PID would resolve stale memory.
type NameCache struct {
	mu    sync.RWMutex
	cache map[nameKey]string
}

type nameKey struct {
	pid uint32
	ptr uint64
}

func NewNameCache() *NameCache {
	return &NameCache{cache: make(map[nameKey]string)}
}

// ForgetPID drops cached resolutions for an exited PID.
func (nc *NameCache) ForgetPID(pid uint32) {
	nc.mu.Lock()
	for k := range nc.cache {
		if k.pid == pid {
			delete(nc.cache, k)
		}
	}
	nc.mu.Unlock()
}

// Sweep drops entries for non-live PIDs (late events repopulate after a
// prune).
func (nc *NameCache) Sweep(live map[int]struct{}) {
	nc.mu.Lock()
	for k := range nc.cache {
		if _, ok := live[int(k.pid)]; !ok {
			delete(nc.cache, k)
		}
	}
	nc.mu.Unlock()
}

func (nc *NameCache) resolveName(pid uint32, ptr uint64) string {
	if ptr == 0 {
		return ""
	}
	key := nameKey{pid: pid, ptr: ptr}
	nc.mu.RLock()
	if n, ok := nc.cache[key]; ok {
		nc.mu.RUnlock()
		return n
	}
	nc.mu.RUnlock()

	rm := remotememory.NewProcessVirtualMemory(libpf.PID(pid))
	name := demangleKernel(rm.String(libpf.Address(ptr)))
	nc.mu.Lock()
	nc.cache[key] = name
	nc.mu.Unlock()
	return name
}

// demangleKernel demangles an Itanium C++ symbol, dropping template and
// parameter detail. No-op for already-clean names.
func demangleKernel(s string) string {
	if !strings.HasPrefix(s, "_Z") {
		return s
	}
	if d := demangle.Filter(s, demangle.NoParams, demangle.NoTemplateParams); d != "" && d != s {
		return d
	}
	return s
}
