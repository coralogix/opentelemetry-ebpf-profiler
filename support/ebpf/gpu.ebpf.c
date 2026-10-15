// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

// Consumers of the otel_gpu USDT probes; the contract is in
// probes/gpu/otel_gpu_usdt.h.

#include "bpfdefs.h"
#include "tracemgmt.h"
#include "types.h"

// Every otel_gpu probe passes its record pointer in this register.
#if defined(__x86_64__)
  #define GPU_RECORD_PTR(ctx) ((ctx)->ax)
#elif defined(__aarch64__)
  #define GPU_RECORD_PTR(ctx) ((ctx)->regs[0])
#else
  #error "unsupported architecture"
#endif

// origin_id_gpu_launch is set at load time by the gpu probe.
BPF_RODATA_VAR(u16, origin_id_gpu_launch, 0)

// Records sent to the gpu probe start with this header; the type selects the
// payload. Layouts are pinned by the probe's decoder.
enum gpu_record_type {
  GPU_RECORD_KERNEL   = 1,
  GPU_RECORD_MEMCPY   = 2,
  GPU_RECORD_EVENT    = 3,
  GPU_RECORD_ERROR    = 4,
  // payload: start ns, end ns, attach cookie of the API function.
  GPU_RECORD_API_SPAN = 5,
};

typedef struct GPURecordHeader {
  u32 type;
  u32 pid;
} GPURecordHeader;

// GPURecord carries a 56-byte producer record verbatim: otel_gpu_kernel_rec,
// otel_gpu_mem_rec or otel_gpu_event_rec.
typedef struct GPURecord {
  GPURecordHeader header;
  u64 payload[7];
} GPURecord;

// GPUErrorRecord carries otel_gpu_error_rec with the message copied in.
typedef struct GPUErrorRecord {
  GPURecordHeader header;
  s32 code;
  u32 _pad;
  char message[120];
} GPUErrorRecord;

_Static_assert(sizeof(GPURecord) == 64, "layout pinned by probes/gpu");
_Static_assert(sizeof(GPUErrorRecord) == 136, "layout pinned by probes/gpu");

// Mirror of struct otel_gpu_error_rec.
struct otel_gpu_error_rec {
  s32 code;
  u32 _pad;
  u64 msg_ptr;
};

// Not among the helpers bpfdefs.h declares for the unwinders.
static long (*bpf_probe_read_user_str)(void *dst, u32 size, const void *unsafe_ptr) = (void *)
  BPF_FUNC_probe_read_user_str;
static u64 (*bpf_get_attach_cookie)(void *ctx) = (void *)BPF_FUNC_get_attach_cookie;

// An API call in progress on a thread; the attach cookie identifies the
// instrumented function.
struct gpu_api_start {
  u64 ts;
  u64 cookie;
};

// gpu_api_starts is created by the gpu probe when API spans are enabled.
struct gpu_api_starts_t {
  __uint(type, BPF_MAP_TYPE_LRU_HASH);
  __type(key, u64);
  __type(value, struct gpu_api_start);
  __uint(max_entries, 16384);
} gpu_api_starts SEC(".maps");

// gpu_events carries GPU records to the gpu probe, which creates it on load.
// Producers flush GPU activity in bursts of tens of thousands of records.
struct gpu_events_t {
  __uint(type, BPF_MAP_TYPE_RINGBUF);
  __uint(max_entries, 1 << 24);
} gpu_events SEC(".maps");

// on_launch: unwind the host stack. The launch record, read whole, is the trace
// value: the correlation id in its low half, the sample weight in its high half.
SEC("uprobe/gpu_launch")
int uprobe__gpu_launch(struct pt_regs *ctx)
{
  u32 pid          = 0;
  u32 tid          = 0;
  u64 group_leader = 0;
  if (!get_pid_tgid_leader(&pid, &tid, &group_leader)) {
    return 0;
  }
  if (pid == 0 || tid == 0) {
    return 0;
  }

  u64 launch = 0;
  if (bpf_probe_read_user(&launch, sizeof(launch), (void *)GPU_RECORD_PTR(ctx))) {
    return 0;
  }
  return collect_trace(
    ctx, origin_id_gpu_launch, pid, tid, group_leader, bpf_ktime_get_ns(), launch);
}

// ready: attached last, its semaphore tells producers that every other probe
// works. There is nothing to record.
SEC("uprobe/gpu_ready")
int uprobe__gpu_ready(UNUSED struct pt_regs *ctx)
{
  return 0;
}

// forward_record sends the probe's record to the gpu probe as the given type.
static EBPF_INLINE int forward_record(struct pt_regs *ctx, u32 type)
{
  u32 pid          = 0;
  u32 tid          = 0;
  u64 group_leader = 0;
  if (!get_pid_tgid_leader(&pid, &tid, &group_leader)) {
    return 0;
  }

  GPURecord record = {.header = {.type = type, .pid = pid}};
  if (bpf_probe_read_user(record.payload, sizeof(record.payload), (void *)GPU_RECORD_PTR(ctx))) {
    return 0;
  }
  bpf_ringbuf_output(&gpu_events, &record, sizeof(record), 0);
  return 0;
}

SEC("uprobe/gpu_kernel_executed")
int uprobe__gpu_kernel_executed(struct pt_regs *ctx)
{
  return forward_record(ctx, GPU_RECORD_KERNEL);
}

SEC("uprobe/gpu_mem")
int uprobe__gpu_mem(struct pt_regs *ctx)
{
  return forward_record(ctx, GPU_RECORD_MEMCPY);
}

SEC("uprobe/gpu_event")
int uprobe__gpu_event(struct pt_regs *ctx)
{
  return forward_record(ctx, GPU_RECORD_EVENT);
}

// error: copy the message here, as the process may exit before the probe
// could read it.
SEC("uprobe/gpu_error")
int uprobe__gpu_error(struct pt_regs *ctx)
{
  u32 pid          = 0;
  u32 tid          = 0;
  u64 group_leader = 0;
  if (!get_pid_tgid_leader(&pid, &tid, &group_leader)) {
    return 0;
  }

  struct otel_gpu_error_rec error;
  if (bpf_probe_read_user(&error, sizeof(error), (void *)GPU_RECORD_PTR(ctx))) {
    return 0;
  }
  GPUErrorRecord record = {
    .header = {.type = GPU_RECORD_ERROR, .pid = pid},
    .code   = error.code,
  };
  if (error.msg_ptr != 0) {
    bpf_probe_read_user_str(record.message, sizeof(record.message), (void *)error.msg_ptr);
  }
  bpf_ringbuf_output(&gpu_events, &record, sizeof(record), 0);
  return 0;
}

// API spans of host GPU libraries such as cuDNN and cuBLAS: the entry time is
// kept per thread and the span is sent when the function returns.
SEC("uprobe/gpu_api_enter")
int uprobe__gpu_api_enter(struct pt_regs *ctx)
{
  u64 pid_tgid               = bpf_get_current_pid_tgid();
  struct gpu_api_start start = {.ts = bpf_ktime_get_ns(), .cookie = bpf_get_attach_cookie(ctx)};
  // Instrumented functions call each other: keep the outermost call.
  bpf_map_update_elem(&gpu_api_starts, &pid_tgid, &start, BPF_NOEXIST);
  return 0;
}

SEC("uprobe/gpu_api_exit")
int uretprobe__gpu_api_exit(struct pt_regs *ctx)
{
  u64 pid_tgid                = bpf_get_current_pid_tgid();
  struct gpu_api_start *start = bpf_map_lookup_elem(&gpu_api_starts, &pid_tgid);
  if (!start || start->cookie != bpf_get_attach_cookie(ctx)) {
    // A nested call returns while the outermost one continues.
    return 0;
  }

  u32 pid          = 0;
  u32 tid          = 0;
  u64 group_leader = 0;
  if (get_pid_tgid_leader(&pid, &tid, &group_leader)) {
    GPURecord record  = {.header = {.type = GPU_RECORD_API_SPAN, .pid = pid}};
    record.payload[0] = start->ts;
    record.payload[1] = bpf_ktime_get_ns();
    record.payload[2] = start->cookie;
    bpf_ringbuf_output(&gpu_events, &record, sizeof(record), 0);
  }
  bpf_map_delete_elem(&gpu_api_starts, &pid_tgid);
  return 0;
}
