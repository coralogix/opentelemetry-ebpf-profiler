// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

// Consumer of the otel_gpu_v1:event USDT probe; the contract is in
// probes/gpu/otel_gpu_v1_usdt.h.

#include "bpfdefs.h"
#include "tracemgmt.h"
#include "types.h"

// The probe passes its event pointer in this register.
#if defined(__x86_64__)
  #define GPU_EVENT_PTR(ctx) ((ctx)->ax)
#elif defined(__aarch64__)
  #define GPU_EVENT_PTR(ctx) ((ctx)->regs[0])
#else
  #error "unsupported architecture"
#endif

// origin_id_gpu_launch is set at load time by the gpu probe.
BPF_RODATA_VAR(u16, origin_id_gpu_launch, 0)

enum {
  // The kind of event whose stack is captured; every other kind is forwarded.
  GPU_EVENT_LAUNCH = 1,
  // The largest payload forwarded, fixed by probes/gpu/otel_gpu_v1_usdt.h.
  GPU_PAYLOAD_MAX  = 4096,
};

// GPUEventHeader is the fixed start of an event, as the producer lays it out.
typedef struct GPUEventHeader {
  u64 correlation_id;
  u32 sample_weight;
  u16 kind;
  u16 payload_size;
} GPUEventHeader;

// GPUEvent is what the gpu probe receives: the PID of the producer's process,
// then its event, of which only the header and payload_size bytes are sent.
typedef struct GPUEvent {
  u32 pid;
  u32 reserved;
  GPUEventHeader header;
  unsigned char payload[GPU_PAYLOAD_MAX];
} GPUEvent;

// gpu_events carries events to the gpu probe, which creates it on load.
// Producers flush GPU activity in bursts of tens of thousands of events.
struct gpu_events_t {
  __uint(type, BPF_MAP_TYPE_RINGBUF);
  __uint(max_entries, 1 << 24);
} gpu_events SEC(".maps");

// gpu_event_scratch holds the event being forwarded, too large for the stack.
struct gpu_event_scratch_t {
  __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
  __type(key, u32);
  __type(value, GPUEvent);
  __uint(max_entries, 1);
} gpu_event_scratch SEC(".maps");

// capture_launch unwinds the host stack of a GPU call. The trace carries the
// call id, which the gpu probe matches GPU work by, and the sample weight.
static EBPF_INLINE int capture_launch(struct pt_regs *ctx, const GPUEventHeader *header)
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
  PerCPURecord *record =
    prepare_trace(origin_id_gpu_launch, pid, tid, group_leader, bpf_ktime_get_boot_ns());
  if (!record) {
    return 0;
  }
  u64 *values = push_context_values(&record->trace, 2);
  if (!values) {
    return 0;
  }
  values[0] = header->correlation_id;
  values[1] = header->sample_weight;
  // A uprobe traps in user space: the kernel stack is only the trap handler.
  return unwind_trace(ctx, record, false);
}

// forward_event sends the event, header and payload, to the gpu probe.
static EBPF_INLINE int forward_event(const GPUEventHeader *header, const unsigned char *event_ptr)
{
  u32 pid          = 0;
  u32 tid          = 0;
  u64 group_leader = 0;
  if (!get_pid_tgid_leader(&pid, &tid, &group_leader)) {
    return 0;
  }
  u32 key         = 0;
  GPUEvent *event = bpf_map_lookup_elem(&gpu_event_scratch, &key);
  if (!event) {
    return 0;
  }

  // Checked right before use, so the verifier knows the bound of size.
  u32 size = header->payload_size;
  if (size > GPU_PAYLOAD_MAX) {
    increment_metric(metricID_BPFGPUEventTooLarge);
    return 0;
  }
  event->pid      = pid;
  event->reserved = 0;
  event->header   = *header;
  if (bpf_probe_read_user(event->payload, size, event_ptr + sizeof(*header))) {
    increment_metric(metricID_BPFGPUEventReadErr);
    return 0;
  }
  if (bpf_ringbuf_output(&gpu_events, event, __builtin_offsetof(GPUEvent, payload) + size, 0) < 0) {
    increment_metric(metricID_BPFGPURingbufOutputErr);
  }
  return 0;
}

SEC("uprobe/gpu_event")
int uprobe__gpu_event(struct pt_regs *ctx)
{
  const unsigned char *event_ptr = (const unsigned char *)GPU_EVENT_PTR(ctx);
  GPUEventHeader header          = {};
  if (bpf_probe_read_user(&header, sizeof(header), event_ptr)) {
    increment_metric(metricID_BPFGPUEventReadErr);
    return 0;
  }
  if (header.kind == GPU_EVENT_LAUNCH) {
    return capture_launch(ctx, &header);
  }
  return forward_event(&header, event_ptr);
}
