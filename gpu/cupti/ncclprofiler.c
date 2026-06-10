//go:build ignore

// SPDX-License-Identifier: Apache-2.0
//
// libotelnccl.so — NCCL profiler plugin (ncclProfiler_v2, NCCL >= 2.24:
// 2.23 introduced the plugin API but its loader resolves ONLY ncclProfiler_v1;
// the v2 fallback chain starts at 2.24 — verified in NCCL's profiler.cc).
// Loaded via NCCL_PROFILER_PLUGIN=/path/to/libotelnccl.so; emits collective
// and point-to-point op spans as otelcupti gpu_event USDT probes, consumed by
// the same eBPF/agent pipeline as the CUPTI shim. Plain C, no NCCL link —
// only the vendored ABI headers (nccl/, Apache-2.0).

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>

#include "nccl/profiler_v2_compat.h"

#include "usdt_probes.h"

struct otel_event {
  uint64_t start_ns;
  const char *func;  // NCCL-owned static string ("AllReduce", ...)
  uint64_t bytes;
};

/* NCCL datatype name → element size; p2p events carry an element count, not
 * bytes (only coll has trafficBytes). NCCL passes its canonical names
 * ("ncclInt8", "ncclFloat16", "ncclBfloat16", ...), which carry the bit
 * width as the digit run — parse that; the digit-less aliases ncclHalf,
 * ncclFloat and ncclDouble are handled explicitly. Unknown future types
 * default to 4. */
static uint64_t datatype_size(const char *dt) {
  if (!dt) {
    return 1;
  }
  for (const char *p = dt; *p; p++) {
    if (*p >= '0' && *p <= '9') {
      int bits = atoi(p);
      return bits >= 8 ? (uint64_t)bits / 8 : 1;
    }
  }
  if (strstr(dt, "Double")) {
    return 8;
  }
  if (strstr(dt, "Half")) {
    return 2;
  }
  return 4; /* ncclFloat / ncclInt / unknown */
}

static uint64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static ncclResult_t otel_init(void **context, int *eActivationMask) {
  *context = NULL;
  *eActivationMask = ncclProfileColl | ncclProfileP2p;
  return ncclSuccess;
}

static ncclResult_t otel_startEvent(void *context, void **eHandle,
                                    ncclProfilerEventDescr_v2_t *eDescr) {
  (void)context;
  *eHandle = NULL;
  if (!eDescr) {
    return ncclSuccess;
  }
  const char *func = NULL;
  uint64_t bytes = 0;
  switch (eDescr->type) {
  case ncclProfileColl:
    func  = eDescr->coll.func;
    bytes = eDescr->coll.trafficBytes;
    break;
  case ncclProfileP2p:
    func  = eDescr->p2p.func;
    bytes = eDescr->p2p.count * datatype_size(eDescr->p2p.datatype);
    break;
  default:
    return ncclSuccess;
  }
  struct otel_event *ev = (struct otel_event *)malloc(sizeof(*ev));
  if (!ev) {
    return ncclSuccess;
  }
  ev->start_ns = now_ns();
  ev->func     = func;
  ev->bytes    = bytes;
  *eHandle = ev;
  return ncclSuccess;
}

static ncclResult_t otel_stopEvent(void *eHandle) {
  struct otel_event *ev = (struct otel_event *)eHandle;
  if (!ev) {
    return ncclSuccess;
  }
  struct otelcupti_event_rec rec = {0};
  rec.kind     = OTELCUPTI_EV_NCCL_OP;
  rec.start    = ev->start_ns;
  rec.end      = now_ns();
  rec.v1       = ev->bytes;
  rec.name_ptr = (uint64_t)(uintptr_t)ev->func;
  OTELCUPTI_GPU_EVENT_REC(&rec);
  free(ev);
  return ncclSuccess;
}

static ncclResult_t otel_recordEventState(void *eHandle,
                                          ncclProfilerEventState_v2_t eState,
                                          ncclProfilerEventStateArgs_v2_t *eStateArgs) {
  (void)eHandle;
  (void)eState;
  (void)eStateArgs;
  return ncclSuccess;
}

static ncclResult_t otel_finalize(void *context) {
  (void)context;
  return ncclSuccess;
}

__attribute__((visibility("default"))) ncclProfiler_v2_t ncclProfiler_v2 = {
  .name             = "otelcupti",
  .init             = otel_init,
  .startEvent       = otel_startEvent,
  .stopEvent        = otel_stopEvent,
  .recordEventState = otel_recordEventState,
  .finalize         = otel_finalize,
};
