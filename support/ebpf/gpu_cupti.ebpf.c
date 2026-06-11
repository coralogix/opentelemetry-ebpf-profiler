// SPDX-License-Identifier: Apache-2.0
//
// gpu_cupti.ebpf.c — consume the otelcupti USDT probes of the CUPTI shim and
// NCCL plugin:
//   on_launch       → collect_trace(TRACE_GPU), correlation id as trace value
//   kernel_executed → cupti_events
//   gpu_mem         → cupti_mem_events
//   gpu_event       → cupti_misc_events (UVM/allocs/memsets/stalls/NCCL)
//   error           → cupti_errors
// plus eBPF-native cuDNN/cuBLAS API spans (api_enter/api_exit uprobe pairs).
// Map names share the cupti_ prefix: the tracer gates GPU map loading on it.

#include "bpfdefs.h"
#include "tracemgmt.h"
#include "types.h"

// Every otelcupti probe passes one struct pointer pinned to a fixed register
// (see usdt_probes.h); the Go side verifies the spec at attach.
#if defined(__x86_64__)
  #define OTELCUPTI_REC_PTR(ctx) ((ctx)->ax)
#elif defined(__aarch64__)
  #define OTELCUPTI_REC_PTR(ctx) ((ctx)->regs[0])
#else
  #error "unsupported architecture for otelcupti USDT consumers"
#endif

// Mirrors of the otelcupti_*_rec structs in usdt_probes.h; keep layouts
// identical.
struct otelcupti_kernel_rec {
  u64 start;
  u64 end;
  u32 correlation_id;
  u32 device_id;
  u32 stream_id;
  u32 graph_id;
  u64 name_ptr;
  u64 nvtx_ptr;
};

struct otelcupti_mem_rec {
  u64 start;
  u64 end;
  u32 correlation_id;
  u32 device_id;
  u32 stream_id;
  u32 copy_kind;
  u64 bytes;
  u64 nvtx_ptr;
};

struct otelcupti_event_rec {
  u64 start;
  u64 end;
  u32 kind;
  u32 correlation_id;
  u64 v1;
  u64 v2;
  u64 name_ptr;
};

struct otelcupti_error_rec {
  s32 code;
  u32 _pad;
  u64 msg_ptr;
};

// 16 MiB: CUPTI flushes whole 8 MiB activity buffers back-to-back, and
// kernel timing is the authoritative total — sized to not drop.
struct cupti_events_t {
  __uint(type, BPF_MAP_TYPE_RINGBUF);
  __uint(max_entries, 1 << 24); // 16 MiB
} cupti_events SEC(".maps");

struct cupti_mem_events_t {
  __uint(type, BPF_MAP_TYPE_RINGBUF);
  __uint(max_entries, 1 << 23); // 8 MiB, same burst reasoning
} cupti_mem_events SEC(".maps");

struct cupti_misc_events_t {
  __uint(type, BPF_MAP_TYPE_RINGBUF);
  __uint(max_entries, 1 << 22); // 4 MiB, low-rate events
} cupti_misc_events SEC(".maps");

struct cupti_errors_t {
  __uint(type, BPF_MAP_TYPE_RINGBUF);
  __uint(max_entries, 1 << 16); // 64 KiB, rare
} cupti_errors SEC(".maps");

// Shim-side error wire record; the message is copied in-probe.
typedef struct GPUShimError {
  s32 code;
  u32 pid;
  char msg[120];
} GPUShimError;

// cuDNN/cuBLAS API spans: entry timestamp keyed by thread, completed on exit.
struct api_start_t {
  u64 ts;
  u64 cookie; // symbol index, set at attach time
};
struct cupti_api_starts_t {
  __uint(type, BPF_MAP_TYPE_LRU_HASH);
  __type(key, u64); // pid_tgid
  __type(value, struct api_start_t);
  __uint(max_entries, 16384);
} cupti_api_starts SEC(".maps");

static void *(*bpf_ringbuf_reserve_)(void *ringbuf, u64 size, u64 flags) = (void *)
  BPF_FUNC_ringbuf_reserve;
static void (*bpf_ringbuf_submit_)(void *data, u64 flags)  = (void *)BPF_FUNC_ringbuf_submit;
static void (*bpf_ringbuf_discard_)(void *data, u64 flags) = (void *)BPF_FUNC_ringbuf_discard;
static long (*bpf_probe_read_user_str_)(void *dst, u32 size, const void *unsafe_ptr) = (void *)
  BPF_FUNC_probe_read_user_str;
static u64 (*bpf_get_attach_cookie_)(void *ctx) = (void *)BPF_FUNC_get_attach_cookie;

// on_launch: capture the host stack, keyed by the correlation id (offset 0
// of otelcupti_launch_rec).
SEC("uprobe/otelcupti_on_launch")
int otel_cupti_on_launch(struct pt_regs *ctx)
{
  u64 pid_tgid = bpf_get_current_pid_tgid();
  u32 pid      = pid_tgid >> 32;
  u32 tid      = pid_tgid & 0xFFFFFFFF;
  if (pid == 0 || tid == 0) {
    return 0;
  }
  u64 recp = OTELCUPTI_REC_PTR(ctx);
  if (recp == 0) {
    return 0;
  }
  u32 corr = 0;
  if (bpf_probe_read_user(&corr, sizeof(corr), (void *)recp)) {
    return 0;
  }
  return collect_trace(ctx, TRACE_GPU, pid, tid, bpf_ktime_get_ns(), corr);
}

// Wire records: the probe's record plus {pid, _pad} at the tail.
struct kernel_executed_wire {
  struct otelcupti_kernel_rec rec;
  u32 pid;
  u32 _pad;
};
struct gpu_mem_wire {
  struct otelcupti_mem_rec rec;
  u32 pid;
  u32 _pad;
};
struct gpu_event_wire {
  struct otelcupti_event_rec rec;
  u32 pid;
  u32 _pad;
};

// Wire sizes are pinned on all three sides (shim asserts, these, Go tests);
// drift anywhere fails a build.
_Static_assert(sizeof(struct kernel_executed_wire) == 56, "pin: gpu/cupti.KernelTiming");
_Static_assert(sizeof(struct gpu_mem_wire) == 56, "pin: gpu/cupti.MemTiming");
_Static_assert(sizeof(struct gpu_event_wire) == 56, "pin: gpu/cupti.GPUEvent");
_Static_assert(sizeof(GPUShimError) == 128, "pin: gpu/cupti.ShimError (seSize)");

// FORWARD_PROBE(name, ringbuf): read the probe's record into a name_##_wire
// and submit it; an unreadable record is discarded outright.
#define FORWARD_PROBE(name_, ringbuf_)                                                             \
  SEC("uprobe/otelcupti_" #name_)                                                                  \
  int otel_cupti_##name_(struct pt_regs *ctx)                                                      \
  {                                                                                                \
    u32 pid  = bpf_get_current_pid_tgid() >> 32;                                                   \
    u64 recp = OTELCUPTI_REC_PTR(ctx);                                                             \
    if (pid == 0 || recp == 0) {                                                                   \
      return 0;                                                                                    \
    }                                                                                              \
    struct name_##_wire *ev = bpf_ringbuf_reserve_(&ringbuf_, sizeof(*ev), 0);                     \
    if (!ev) {                                                                                     \
      return 0;                                                                                    \
    }                                                                                              \
    if (bpf_probe_read_user(&ev->rec, sizeof(ev->rec), (void *)recp)) {                            \
      bpf_ringbuf_discard_(ev, 0);                                                                 \
      return 0;                                                                                    \
    }                                                                                              \
    ev->pid  = pid;                                                                                \
    ev->_pad = 0;                                                                                  \
    bpf_ringbuf_submit_(ev, 0);                                                                    \
    return 0;                                                                                      \
  }

FORWARD_PROBE(kernel_executed, cupti_events)
FORWARD_PROBE(gpu_mem, cupti_mem_events)
FORWARD_PROBE(gpu_event, cupti_misc_events)

// error: copy the static message in-probe — the Go side must not need a
// /proc/<pid>/mem read from a process that may be exiting.
SEC("uprobe/otelcupti_error")
int otel_cupti_error(struct pt_regs *ctx)
{
  u32 pid  = bpf_get_current_pid_tgid() >> 32;
  u64 recp = OTELCUPTI_REC_PTR(ctx);
  if (pid == 0 || recp == 0) {
    return 0;
  }
  struct otelcupti_error_rec r = {};
  if (bpf_probe_read_user(&r, sizeof(r), (void *)recp)) {
    return 0;
  }
  GPUShimError *ev = bpf_ringbuf_reserve_(&cupti_errors, sizeof(*ev), 0);
  if (!ev) {
    return 0;
  }
  ev->code   = r.code;
  ev->pid    = pid;
  ev->msg[0] = 0;
  if (r.msg_ptr != 0) {
    (void)bpf_probe_read_user_str_(ev->msg, sizeof(ev->msg), (void *)r.msg_ptr);
  }
  bpf_ringbuf_submit_(ev, 0);
  return 0;
}

// Mirror of OTELCUPTI_EV_API_SPAN (usdt_probes.h) / EvAPISpan (Go).
#define GPU_EVENT_KIND_API 11

SEC("uprobe/otelcupti_api_enter")
int otel_cupti_api_enter(struct pt_regs *ctx)
{
  u64 pid_tgid = bpf_get_current_pid_tgid();
  if ((pid_tgid >> 32) == 0) {
    return 0;
  }
  struct api_start_t st;
  st.ts     = bpf_ktime_get_ns();
  st.cookie = bpf_get_attach_cookie_(ctx);
  // NOEXIST: instrumented APIs nest; keep the outermost span (the exit pairs
  // by cookie). Best-effort under LRU eviction.
  bpf_map_update_elem(&cupti_api_starts, &pid_tgid, &st, BPF_NOEXIST);
  return 0;
}

SEC("uprobe/otelcupti_api_exit")
int otel_cupti_api_exit(struct pt_regs *ctx)
{
  (void)ctx;
  u64 pid_tgid = bpf_get_current_pid_tgid();
  u32 pid      = pid_tgid >> 32;
  if (pid == 0) {
    return 0;
  }
  struct api_start_t *st = bpf_map_lookup_elem(&cupti_api_starts, &pid_tgid);
  if (!st || st->cookie != bpf_get_attach_cookie_(ctx)) {
    return 0; // no entry, or this is a nested call's exit — keep the outer span
  }
  struct gpu_event_wire *ev = bpf_ringbuf_reserve_(&cupti_misc_events, sizeof(*ev), 0);
  if (!ev) {
    bpf_map_delete_elem(&cupti_api_starts, &pid_tgid);
    return 0;
  }
  ev->rec       = (struct otelcupti_event_rec){};
  ev->rec.start = st->ts;
  ev->rec.end   = bpf_ktime_get_ns();
  ev->rec.kind  = GPU_EVENT_KIND_API;
  ev->rec.v1    = st->cookie;
  ev->pid       = pid;
  ev->_pad      = 0;
  bpf_ringbuf_submit_(ev, 0);
  bpf_map_delete_elem(&cupti_api_starts, &pid_tgid);
  return 0;
}
