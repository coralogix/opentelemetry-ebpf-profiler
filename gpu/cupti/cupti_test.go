// SPDX-License-Identifier: Apache-2.0

package cupti

import (
	"encoding/binary"
	"testing"
	"unsafe"
)

// Wire sizes of the eBPF ringbuf records (FORWARD_PROBE: rec + pid/pad tail).
// Each pin asserts binary.Size == unsafe.Sizeof == wire size: equality proves
// the struct has no implicit padding, so pfunsafe.Read's memory-layout cast
// matches the packed C record.
const (
	ktSize = 56
	mtSize = 56
	geSize = 56
)

func assertWire[T any](t *testing.T, want int) {
	t.Helper()
	var v T
	if got := binary.Size(v); got != want {
		t.Fatalf("binary.Size = %d, want %d", got, want)
	}
	if got := int(unsafe.Sizeof(v)); got != want {
		t.Fatalf("unsafe.Sizeof = %d, want %d (implicit padding breaks pfunsafe.Read)", got, want)
	}
}

func TestKernelTimingWireSize(t *testing.T) { assertWire[KernelTiming](t, ktSize) }

func TestMemTimingWireSize(t *testing.T) { assertWire[MemTiming](t, mtSize) }

func TestGPUEventWireSize(t *testing.T) { assertWire[GPUEvent](t, geSize) }

func TestForgetPID(t *testing.T) {
	nc := NewNameCache()
	nc.cache[nameKey{pid: 1, ptr: 0x10}] = "a"
	nc.cache[nameKey{pid: 1, ptr: 0x20}] = "b"
	nc.cache[nameKey{pid: 2, ptr: 0x10}] = "c"

	nc.ForgetPID(1)

	if _, ok := nc.cache[nameKey{pid: 1, ptr: 0x10}]; ok {
		t.Error("pid 1 entry survived ForgetPID")
	}
	if _, ok := nc.cache[nameKey{pid: 1, ptr: 0x20}]; ok {
		t.Error("pid 1 entry survived ForgetPID")
	}
	if _, ok := nc.cache[nameKey{pid: 2, ptr: 0x10}]; !ok {
		t.Error("pid 2 entry wrongly dropped")
	}
}
