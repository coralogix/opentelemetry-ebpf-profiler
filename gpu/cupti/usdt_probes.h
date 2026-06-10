// SPDX-License-Identifier: Apache-2.0
//
// usdt_probes.h — the USDT contract between the CUDA injection shim
// (libotelcupti.so) and the profiler's eBPF consumers in
// support/ebpf/gpu_cupti.ebpf.c.
//
// Provider "otelcupti". Every probe passes a SINGLE struct pointer, pinned to
// a fixed register via a local register-asm variable (honored by GCC and
// Clang when the variable is used as an asm operand, which is what sys/sdt.h
// does — and asserted on the build output by `make check`, the authority). The
// .note.stapsdt arg spec is therefore exactly "8@%rax" (amd64) / "8@x0"
// (arm64) on every build, so the eBPF side can read the pointer from a known
// register. Without pinning, the compiler picks a register per call site —
// and multi-arg probes produce memory operands — neither decodable
// build-stably. `make check` asserts the spec at build time; the agent
// re-verifies it at attach time.
//
// Struct layouts MUST match the mirrors in gpu_cupti.ebpf.c.
#pragma once

#include <stddef.h>
#include <stdint.h>
#include <sys/sdt.h>

#if defined(__x86_64__)
  #define OTELCUPTI_REC_REG "rax"
#elif defined(__aarch64__)
  #define OTELCUPTI_REC_REG "x0"
#else
  #error "unsupported architecture for otelcupti USDT probes"
#endif
#define OTELCUPTI_PROBE1_PINNED(name, ptr)                              \
  do {                                                                  \
    register void *otelcupti_rec_ asm(OTELCUPTI_REC_REG) = (void *)(ptr); \
    DTRACE_PROBE1(otelcupti, name, otelcupti_rec_);                     \
  } while (0)

// CPU side: fired from a CUPTI enter-callback at each tracked call site
// (kernel launches and memcpys). The profiler attaches collect_trace() here
// to capture the host stack, keyed by correlation id.
struct otelcupti_launch_rec {
  uint32_t correlation_id; // MUST stay first: the eBPF reads it at offset 0
  uint32_t cbid;           // CUPTI callback id of the call site
  uint64_t name_ptr;       // const char* launch symbol (0 if none)
};
_Static_assert(offsetof(struct otelcupti_launch_rec, correlation_id) == 0,
               "eBPF on_launch reads the correlation id at offset 0");
#define OTELCUPTI_ON_LAUNCH_REC(recptr) \
  OTELCUPTI_PROBE1_PINNED(on_launch, (recptr))

// GPU side: real kernel execution window from CUPTI Activity.
struct otelcupti_kernel_rec {
  uint64_t start;          // GPU HW-counter start (ns)
  uint64_t end;            // GPU HW-counter end (ns)
  uint32_t correlation_id; // joins to the on_launch host stack
  uint32_t device_id;
  uint32_t stream_id;
  uint32_t graph_id;       // CUDA graph id (0 = not graph-launched)
  uint64_t name_ptr;       // const char* kernel name (read via /proc/pid/mem)
  uint64_t nvtx_ptr;       // const char* NVTX range nesting at launch (0 = none)
};
#define OTELCUPTI_KERNEL_EXECUTED_REC(recptr) \
  OTELCUPTI_PROBE1_PINNED(kernel_executed, (recptr))

// GPU memcpy activity record.
struct otelcupti_mem_rec {
  uint64_t start;
  uint64_t end;
  uint32_t correlation_id; // joins to the on_launch host stack
  uint32_t device_id;
  uint32_t stream_id;
  uint32_t copy_kind;      // CUpti_ActivityMemcpyKind (1=HtoD, 2=DtoH, ...)
  uint64_t bytes;
  uint64_t nvtx_ptr;       // const char* NVTX range nesting at the call (0 = none)
};
#define OTELCUPTI_GPU_MEM_REC(recptr) \
  OTELCUPTI_PROBE1_PINNED(gpu_mem, (recptr))

// Shim-side error (e.g. another CUPTI subscriber present). Re-emitted
// periodically: the initial cuInit-time emission always races the agent's
// USDT attach.
struct otelcupti_error_rec {
  int32_t code;     // CUptiResult or -1 for shim-internal failures
  uint32_t _pad;
  uint64_t msg_ptr; // const char* static message
};
#define OTELCUPTI_ERROR_REC(recptr) \
  OTELCUPTI_PROBE1_PINNED(error, (recptr))

// Generic low-rate GPU event, discriminated by kind. Shared by UVM counters,
// allocations, memsets, PC-sampling stall histograms and the NCCL profiler
// plugin (libotelnccl.so emits it too). Field semantics per kind below.
enum otelcupti_event_kind {
  OTELCUPTI_EV_UVM_HTOD    = 1,  // v1 = bytes migrated host→device
  OTELCUPTI_EV_UVM_DTOH    = 2,  // v1 = bytes migrated device→host
  OTELCUPTI_EV_UVM_CPU_FLT = 3,  // v1 = CPU page faults
  OTELCUPTI_EV_UVM_GPU_FLT = 4,  // v1 = GPU page fault groups
  OTELCUPTI_EV_ALLOC       = 5,  // v1 = bytes, v2 = CUpti memoryKind, corr set
  OTELCUPTI_EV_FREE        = 6,  // v1 = bytes, v2 = CUpti memoryKind, corr set
  OTELCUPTI_EV_MEMSET      = 7,  // v1 = bytes, corr set, start/end = GPU ns
  OTELCUPTI_EV_STALL       = 8,  // name_ptr = kernel func, v1 = const char* stall
                                 // reason, v2 = sample count
  OTELCUPTI_EV_NCCL_OP     = 10, // name_ptr = op name, v1 = traffic bytes,
                                 // start/end = CLOCK_MONOTONIC ns
  OTELCUPTI_EV_API_SPAN    = 11, // cuDNN/cuBLAS API spans. Emitted by the
                                 // eBPF api_exit program (no USDT emission);
                                 // declared here so the kind namespace has a
                                 // single owner. Mirrors GPU_EVENT_KIND_API
                                 // in gpu_cupti.ebpf.c and EvAPISpan in Go.
};
struct otelcupti_event_rec {
  uint64_t start;
  uint64_t end;
  uint32_t kind; // enum otelcupti_event_kind
  uint32_t correlation_id;
  uint64_t v1;
  uint64_t v2;
  uint64_t name_ptr;
};

// The eBPF wire structs append {u32 pid, u32 pad}: Go decodes these layouts
// + 8 at fixed offsets (gpu/cupti wire tests), so the record sizes are part
// of the contract.
_Static_assert(sizeof(struct otelcupti_launch_rec) == 16, "wire contract");
_Static_assert(sizeof(struct otelcupti_kernel_rec) == 48, "wire contract");
_Static_assert(sizeof(struct otelcupti_mem_rec) == 48, "wire contract");
_Static_assert(sizeof(struct otelcupti_error_rec) == 16, "wire contract");
_Static_assert(sizeof(struct otelcupti_event_rec) == 48, "wire contract");
#define OTELCUPTI_GPU_EVENT_REC(recptr) \
  OTELCUPTI_PROBE1_PINNED(gpu_event, (recptr))
