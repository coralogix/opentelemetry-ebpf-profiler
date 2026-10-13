// SPDX-License-Identifier: Apache-2.0
//
// usdt_producer — bridge proof. Fires the otelcupti USDT probes in a loop
// with NO CUDA/CUPTI/GPU, so the profiler-side USDT attach + decode path can
// be validated standalone.
//
// Build:  make -C gpu/cupti/test usdt_producer   (needs systemtap-sdt-dev)
// Run:    ./usdt_producer
// Verify: sudo bpftrace -e 'usdt:./usdt_producer:otelcupti:kernel_executed { printf("%lx\n", arg0); }'
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

#include "../usdt_probes.h"

int main(void) {
  uint32_t corr = 0;
  const char *names[] = {"mul_mat_vec_q", "flash_attn_ext_f16", "rms_norm_f32"};
  fprintf(stderr, "usdt_producer: firing otelcupti probes (pid %d)\n", getpid());
  for (;;) {
    corr++;
    const char *name = names[corr % 3];

    struct otelcupti_launch_rec lrec = {
      .correlation_id = corr,
      .cbid           = 0,
      .name_ptr       = (uint64_t)(uintptr_t)name,
    };
    OTELCUPTI_ON_LAUNCH_REC(&lrec);

    uint64_t start = (uint64_t)corr * 1000;
    struct otelcupti_kernel_rec krec = {
      .start          = start,
      .end            = start + (100 + corr % 900),
      .correlation_id = corr,
      .device_id      = 0,
      .stream_id      = 7,
      .graph_id       = 0,
      .name_ptr       = (uint64_t)(uintptr_t)name,
      .nvtx_ptr       = 0,
    };
    OTELCUPTI_KERNEL_EXECUTED_REC(&krec);

    struct timespec ts = {0, 5 * 1000 * 1000}; // 5ms
    nanosleep(&ts, NULL);
  }
  return 0;
}
