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
