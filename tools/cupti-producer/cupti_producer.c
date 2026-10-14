// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0
//
// libotelcupti.so — CUDA injection library (CUDA_INJECTION64_PATH), a
// reference producer of the otel_gpu USDT probes consumed by probes/gpu.
// Captures GPU timing and correlation ids through CUPTI and fires them as
// probes of provider "otel_gpu" (probes/gpu/otel_gpu_usdt.h).
//
// Plain C on purpose: the shim loads into arbitrary host processes, so no
// libstdc++/C++ runtime ABI exposure (demangling happens on the Go side).
//
// Coexistence: CUPTI allows one subscriber per process before CUDA 13.2 /
// r610. If another client (Kineto, Nsight, DCGM) holds it, we emit an
// `error` probe and disable gracefully.

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <cuda.h> /* CUDA_VERSION */
#include <cupti.h>
#include <cupti_pcsampling.h>
#include <nvtx3/nvToolsExt.h> /* nvtxEventAttributes_t for NVTX range messages (NVTX3) */

#if CUDA_VERSION < 12000
  #error "CUDA 12 or newer is required"
#endif

static void pc_on_context_created(CUcontext ctx);
static void pc_on_context_destroy(CUcontext ctx);
static void uvm_configure_ctx(CUcontext ctx);

#define OTEL_GPU_SEMAPHORES
#include "otel_gpu_usdt.h"
#include "sampler.h"

// CUPTI fills kernel activity records as the LATEST CUpti_ActivityKernelN
// the shipped libcupti supports — NOT the unversioned alias, which
// cupti_activity_deprecated.h pins to an old layout with different field
// offsets (casting to it reads garbage). Pick the version matching the
// toolkit we build and ship against.
#if CUDA_VERSION >= 12040
typedef CUpti_ActivityKernel9 otelcupti_kernel_activity_t;
#else
typedef CUpti_ActivityKernel8 otelcupti_kernel_activity_t;
#endif

#define OTELCUPTI_BUF_SIZE (8 * 1024 * 1024) /* 8 MiB activity buffer */
#define OTELCUPTI_ALIGN    8

static CUpti_SubscriberHandle g_subscriber;
static int g_enabled;

/* An error fired at cuInit races the profiler's USDT attach, so keep the
 * last error and re-emit it periodically and at exit. msg is static. */
static pthread_mutex_t g_err_mu = PTHREAD_MUTEX_INITIALIZER;
static struct otel_gpu_error_rec g_last_err;
static int g_err_started;
static void start_error_reemit(void);

/* The mutex keeps the re-emitted (code, msg) pair consistent. Rare path. */
static void emit_error(int code, const char *msg)
{
  struct otel_gpu_error_rec rec = {
    .code    = code,
    ._pad    = 0,
    .msg_ptr = (uint64_t)(uintptr_t)msg,
  };
  OTEL_GPU_PROBE(error, &rec);
  pthread_mutex_lock(&g_err_mu);
  g_last_err = rec;
  if (!g_err_started) {
    g_err_started = 1;
    start_error_reemit();
  }
  pthread_mutex_unlock(&g_err_mu);
}

static void *error_reemit_thread(void *arg)
{
  (void)arg;
  unsigned interval = 60;
  const char *s     = getenv("OTEL_GPU_ERROR_REEMIT_S");
  if (s && atoi(s) > 0) {
    interval = (unsigned)atoi(s);
  }
  for (;;) {
    sleep(interval);
    pthread_mutex_lock(&g_err_mu);
    struct otel_gpu_error_rec rec = g_last_err;
    pthread_mutex_unlock(&g_err_mu);
    OTEL_GPU_PROBE(error, &rec);
  }
  return NULL;
}

static void start_error_reemit(void)
{
  pthread_t t;
  pthread_attr_t a;
  pthread_attr_init(&a);
  pthread_attr_setdetachstate(&a, PTHREAD_CREATE_DETACHED);
  pthread_create(&t, &a, error_reemit_thread, NULL);
  pthread_attr_destroy(&a);
}

/* ─── NVTX range tracking ────────────────────────────────────────────────────
 * NVTX ranges carry app-level names stripped host stacks lack. Keep a
 * per-thread range stack, intern the names, and keep the active nesting per
 * call site so buffer_completed can attach it per kernel. */
#define NVTX_MAX_DEPTH 64
#define NVTX_POOL_MAX  8192

/* Intern: dedup + strdup, NEVER freed — the agent reads the strings from
 * /proc/<pid>/mem asynchronously. Insert-only open-addressed hash: lock-free
 * reads, inserts take the mutex and re-probe. */
static pthread_mutex_t g_nvtx_pool_mu = PTHREAD_MUTEX_INITIALIZER;
static const char *g_nvtx_pool[NVTX_POOL_MAX]; /* power-of-two slots */

static uint32_t fnv1a(const char *s)
{
  uint32_t h = 2166136261u;
  for (; *s; s++) {
    h = (h ^ (uint8_t)*s) * 16777619u;
  }
  return h;
}

static const char *nvtx_intern(const char *s)
{
  if (!s || !*s) {
    return NULL;
  }
  uint32_t h = fnv1a(s);
  for (uint32_t i = 0; i < NVTX_POOL_MAX; i++) {
    uint32_t slot = (h + i) & (NVTX_POOL_MAX - 1);
    const char *p = __atomic_load_n(&g_nvtx_pool[slot], __ATOMIC_ACQUIRE);
    if (p == NULL) {
      pthread_mutex_lock(&g_nvtx_pool_mu);
      /* Re-probe under the lock: another thread may have inserted. */
      const char *found = NULL;
      for (; i < NVTX_POOL_MAX; i++) {
        slot = (h + i) & (NVTX_POOL_MAX - 1);
        p    = g_nvtx_pool[slot];
        if (p == NULL) {
          found = strdup(s);
          __atomic_store_n(&g_nvtx_pool[slot], found, __ATOMIC_RELEASE);
          break;
        }
        if (strcmp(p, s) == 0) {
          found = p;
          break;
        }
      }
      pthread_mutex_unlock(&g_nvtx_pool_mu);
      return found; /* NULL when the table filled up concurrently */
    }
    if (strcmp(p, s) == 0) {
      return p;
    }
  }
  return NULL; /* table full */
}

static __thread const char *t_nvtx_stack[NVTX_MAX_DEPTH];
static __thread int t_nvtx_depth;
/* Active range nesting as one interned string, innermost-first,
 * '\x1f'-separated. Recomputed at push/pop; launches just read it. */
static __thread const char *t_nvtx_joined;

#define NVTX_SEP '\x1f'
static void nvtx_recompute_joined(void)
{
  t_nvtx_joined = NULL;
  if (t_nvtx_depth <= 0) {
    return;
  }
  int n = t_nvtx_depth < NVTX_MAX_DEPTH ? t_nvtx_depth : NVTX_MAX_DEPTH;
  static __thread char buf[1024];
  size_t off = 0;
  for (int i = n - 1; i >= 0; i--) { /* innermost first */
    const char *s = t_nvtx_stack[i];
    if (!s) {
      continue;
    }
    size_t l = strlen(s);
    if (off + l + 2 >= sizeof(buf)) {
      break;
    }
    if (off) {
      buf[off++] = NVTX_SEP;
    }
    memcpy(buf + off, s, l);
    off += l;
  }
  if (off > 0) {
    buf[off]      = 0;
    t_nvtx_joined = nvtx_intern(buf);
  }
}
static void nvtx_thread_push(const char *interned)
{
  /* depth is never negative: pop only decrements when > 0 */
  if (t_nvtx_depth < NVTX_MAX_DEPTH) {
    t_nvtx_stack[t_nvtx_depth] = interned;
  }
  t_nvtx_depth++; /* count past MAX so pop stays balanced */
  nvtx_recompute_joined();
}
static void nvtx_thread_pop(void)
{
  if (t_nvtx_depth > 0) {
    t_nvtx_depth--;
  }
  nvtx_recompute_joined();
}

/* ─── Call sites by correlation id ───────────────────────────────────────────
 * Every tracked call site leaves its sample weight (0 if not sampled) and its
 * NVTX nesting here, for buffer_completed to filter and label its GPU records.
 * Direct-mapped on sequential correlation ids = FIFO of the last SLOTS ids.
 * Must exceed the ids issued between a call and the flush of its records
 * (CUPTI flushes every second). Zero-page BSS. */
#define CALL_SITE_SLOTS (1 << 19) /* power of two */

/* correlation id << 32 | sample weight; packed so a reader sees both or neither.
 * A torn read of the NVTX nesting only mislabels a kernel. */
static uint64_t g_site_key[CALL_SITE_SLOTS];
static const char *g_site_nvtx[CALL_SITE_SLOTS];

static void remember_call_site(uint32_t corr, uint32_t weight, const char *nvtx)
{
  if (corr == 0) {
    return;
  }
  uint32_t h = corr & (CALL_SITE_SLOTS - 1);
  __atomic_store_n(&g_site_nvtx[h], nvtx, __ATOMIC_RELEASE);
  __atomic_store_n(&g_site_key[h], (uint64_t)corr << 32 | weight, __ATOMIC_RELEASE);
}

static void report_lost_call_sites(void)
{
  static int reported;
  if (!__atomic_exchange_n(&reported, 1, __ATOMIC_RELAXED)) {
    emit_error(-1, "call site table too small: GPU records were dropped");
  }
}

/* Returns the sample weight of the GPU records of call corr, 0 to drop them,
 * and its NVTX nesting. Records of untracked calls count once. */
static uint32_t call_site_weight(uint32_t corr, const char **nvtx)
{
  *nvtx = NULL;
  if (corr == 0) {
    return 1;
  }
  uint32_t h          = corr & (CALL_SITE_SLOTS - 1);
  const uint64_t key  = __atomic_load_n(&g_site_key[h], __ATOMIC_ACQUIRE);
  const uint32_t site = (uint32_t)(key >> 32);
  if (key != 0 && site == corr) {
    *nvtx = __atomic_load_n(&g_site_nvtx[h], __ATOMIC_ACQUIRE);
    return (uint32_t)key;
  }
  if (key != 0 && (int32_t)(site - corr) > 0) {
    /* A later call took the slot: whether this one was sampled is lost. */
    report_lost_call_sites();
    return 0;
  }
  return 1;
}

/* ─── Call site sampling ────────────────────────────────────────────────────
 * Kernel launches, graph launches, memory operations and allocations are
 * sampled apart, so a rare class keeps its samples next to a frequent one. */
enum call_site_class {
  SITE_NONE = -1,
  SITE_KERNEL,
  SITE_GRAPH,
  SITE_MEMORY,
  SITE_ALLOC,
  SITE_CLASSES,
};

/* Default sampled call sites per thread per second. Graph launches are sampled
 * less, as each runs many kernels. */
#define DEFAULT_SAMPLE_RATE       200
#define DEFAULT_GRAPH_SAMPLE_RATE 20

/* Sampled call sites per thread per second, by class; 0 samples all of them. */
static uint32_t g_sample_rate[SITE_CLASSES];

static __thread struct otel_sampler t_samplers[SITE_CLASSES];
static __thread uint64_t t_rng;

static uint64_t now_ns(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static uint32_t sample_call_site(enum call_site_class cls)
{
  const uint64_t now = now_ns();
  if (t_rng == 0) {
    t_rng = (now ^ (uint64_t)(uintptr_t)&t_rng) | 1;
  }
  return otel_sampler_weight(&t_samplers[cls], g_sample_rate[cls], now, &t_rng);
}

/* Reads a sampling rate from the environment; fallback if unset or invalid. */
static uint32_t env_rate(const char *name, uint32_t fallback)
{
  const char *s = getenv(name);
  if (!s || !*s) {
    return fallback;
  }
  char *end;
  const unsigned long v = strtoul(s, &end, 10);
  return (*end == '\0' && v <= UINT32_MAX) ? (uint32_t)v : fallback;
}

/* Extract the message from an NVTX push callback's params. The param layout
 * is CUPTI's undocumented injection ABI, so match function names EXACTLY —
 * a substring match against a future variant with different args would read
 * a wild pointer. Unmatched pushes stay unnamed but still balance the pop. */
static const char *nvtx_push_message(const char *fn, const void *params)
{
  if (!fn || !params) {
    return NULL;
  }
  if (strcmp(fn, "nvtxRangePushA") == 0) {
    /* params -> { const char* message } */
    return *(const char *const *)params;
  }
  if (strcmp(fn, "nvtxRangePushEx") == 0 || strcmp(fn, "nvtxDomainRangePushEx") == 0) {
    /* params -> { [nvtxDomainHandle_t domain,] nvtxEventAttributes_t* } */
    const nvtxEventAttributes_t *ea;
    if (fn[4] == 'D') { /* nvtxDomain... */
      ea = *(const nvtxEventAttributes_t *const *)((const char *)params + sizeof(void *));
    } else {
      ea = *(const nvtxEventAttributes_t *const *)params;
    }
    if (ea && ea->messageType == NVTX_MESSAGE_TYPE_ASCII) {
      return ea->message.ascii;
    }
  }
  return NULL; /* PushW (wide) or unknown → unnamed, still balanced */
}

/* CUPTI NVTX-domain callback: maintain the per-thread range stack. */
static void handle_nvtx(const CUpti_NvtxData *d)
{
  if (!d || !d->functionName) {
    return;
  }
  const char *fn = d->functionName;
  if (strstr(fn, "RangePop")) {
    nvtx_thread_pop();
  } else if (strstr(fn, "RangePush")) {
    const char *msg      = nvtx_push_message(fn, d->functionParams);
    const char *interned = NULL;
    if (msg) {
      // Drop PyTorch's per-call ", seq = N, ..." suffix so identical ops
      // aggregate and the intern pool stays bounded by distinct names.
      char clean[128];
      size_t i = 0;
      for (; msg[i] && i < sizeof(clean) - 1; i++) {
        if (msg[i] == ',' && msg[i + 1] == ' ') {
          break;
        }
        clean[i] = msg[i];
      }
      clean[i] = 0;
      interned = nvtx_intern(clean);
    }
    nvtx_thread_push(interned);
  }
}

/* ─── Activity API: real GPU timing ─────────────────────────────────────────── */
static void CUPTIAPI buffer_requested(uint8_t **buffer, size_t *size, size_t *max_num_records)
{
  void *p = NULL;
  if (posix_memalign(&p, OTELCUPTI_ALIGN, OTELCUPTI_BUF_SIZE) != 0) {
    p     = NULL;
    *size = 0;
  } else {
    *size = OTELCUPTI_BUF_SIZE;
  }
  *buffer          = (uint8_t *)p; /* freed (same pointer) in buffer_completed */
  *max_num_records = 0;            /* fill as many as fit */
}

static void CUPTIAPI
buffer_completed(CUcontext ctx, uint32_t stream_id, uint8_t *buffer, size_t size, size_t valid_size)
{
  (void)ctx;
  (void)stream_id;
  (void)size;

  CUpti_Activity *record = NULL;
  CUptiResult st;
  do {
    st = cuptiActivityGetNextRecord(buffer, valid_size, &record);
    if (st == CUPTI_SUCCESS) {
      switch (record->kind) {
      case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
      case CUPTI_ACTIVITY_KIND_KERNEL: {
        otelcupti_kernel_activity_t *k = (otelcupti_kernel_activity_t *)record;
        const char *nvtx;
        const uint32_t weight = call_site_weight(k->correlationId, &nvtx);
        if (weight == 0) {
          break;
        }
        struct otel_gpu_kernel_rec rec = {0};
        rec.start                      = k->start;
        rec.end                        = k->end;
        rec.correlation_id             = k->correlationId;
        rec.device_id                  = k->deviceId;
        rec.stream_id                  = k->streamId;
        rec.graph_id                   = k->graphId;
        rec.name_ptr                   = (uint64_t)(uintptr_t)(k->name ? k->name : "");
        rec.ranges_ptr                 = (uint64_t)(uintptr_t)nvtx;
        rec.sample_weight              = weight;
        OTEL_GPU_PROBE(kernel_executed, &rec);
        break;
      }
      case CUPTI_ACTIVITY_KIND_MEMCPY: {
        CUpti_ActivityMemcpy *m = (CUpti_ActivityMemcpy *)record;
        const char *nvtx;
        const uint32_t weight = call_site_weight(m->correlationId, &nvtx);
        if (weight == 0) {
          break;
        }
        struct otel_gpu_mem_rec rec = {0};
        rec.start                   = m->start;
        rec.end                     = m->end;
        rec.correlation_id          = m->correlationId;
        rec.device_id               = m->deviceId;
        rec.stream_id               = m->streamId;
        rec.copy_kind               = m->copyKind;
        rec.bytes                   = m->bytes;
        rec.ranges_ptr              = (uint64_t)(uintptr_t)nvtx;
        rec.sample_weight           = weight;
        OTEL_GPU_PROBE(gpu_mem, &rec);
        break;
      }
      case CUPTI_ACTIVITY_KIND_MEMSET: {
        CUpti_ActivityMemset *m = (CUpti_ActivityMemset *)record;
        const char *nvtx;
        const uint32_t weight = call_site_weight(m->correlationId, &nvtx);
        if (weight == 0) {
          break;
        }
        struct otel_gpu_event_rec ev = {0};
        ev.kind                      = OTEL_GPU_EVENT_MEMSET;
        ev.start                     = m->start;
        ev.end                       = m->end;
        ev.correlation_id            = m->correlationId;
        ev.v1                        = m->bytes;
        ev.sample_weight             = weight;
        OTEL_GPU_PROBE(gpu_event, &ev);
        break;
      }
      case CUPTI_ACTIVITY_KIND_MEMORY2: {
        CUpti_ActivityMemory3 *m = (CUpti_ActivityMemory3 *)record;
        const char *nvtx;
        const uint32_t weight = call_site_weight(m->correlationId, &nvtx);
        if (weight == 0) {
          break;
        }
        struct otel_gpu_event_rec ev = {0};
        ev.kind  = (m->memoryOperationType == CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_ALLOCATION)
                     ? OTEL_GPU_EVENT_ALLOC
                     : OTEL_GPU_EVENT_FREE;
        ev.start = m->timestamp;
        ev.end   = m->timestamp;
        ev.correlation_id = m->correlationId;
        ev.v1             = m->bytes;
        ev.v2             = m->memoryKind;
        ev.sample_weight  = weight;
        OTEL_GPU_PROBE(gpu_event, &ev);
        break;
      }
      case CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER: {
        CUpti_ActivityUnifiedMemoryCounter2 *u = (CUpti_ActivityUnifiedMemoryCounter2 *)record;
        struct otel_gpu_event_rec ev           = {0};
        switch (u->counterKind) {
        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD:
          ev.kind = OTEL_GPU_EVENT_UVM_HTOD;
          break;
        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH:
          ev.kind = OTEL_GPU_EVENT_UVM_DTOH;
          break;
        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT:
          ev.kind = OTEL_GPU_EVENT_UVM_CPU_FLT;
          ev.v2   = 1; // value carries the faulting address, not a count
          break;
        case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT:
          ev.kind = OTEL_GPU_EVENT_UVM_GPU_FLT;
          break;
        default: ev.kind = 0; break;
        }
        if (ev.kind != 0) {
          ev.start = u->start;
          ev.end   = u->end;
          ev.v1    = (ev.v2 != 0) ? ev.v2 : u->value; // v2 set = per-record count
          ev.v2    = 0;
          OTEL_GPU_PROBE(gpu_event, &ev);
        }
        break;
      }
      default: break;
      }
    } else if (st != CUPTI_ERROR_MAX_LIMIT_REACHED) {
      break;
    }
  } while (st == CUPTI_SUCCESS);

  free(buffer); /* posix_memalign'd in buffer_requested → same pointer */
}

/* ─── Callback API: launch correlation (CPU side) ──────────────────────────── */

/* Tracked call sites: their callbacks are enabled and they fire on_launch.
 * ggml/llama.cpp and most native backends launch through the driver API. */
static const struct {
  CUpti_CallbackDomain domain;
  CUpti_CallbackId cbid;
  enum call_site_class cls;
} g_call_sites[] = {
  {CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000, SITE_KERNEL},
  {CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060, SITE_KERNEL},
  {CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_v10000, SITE_GRAPH},
  {CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020, SITE_MEMORY},
  {CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020, SITE_MEMORY},
  {CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaMemset_v3020, SITE_MEMORY},
  {CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaMemsetAsync_v3020, SITE_MEMORY},
  {CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020, SITE_ALLOC},
  {CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaFree_v3020, SITE_ALLOC},
  {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel, SITE_KERNEL},
  {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz, SITE_KERNEL},
  {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx, SITE_KERNEL},
  {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz, SITE_KERNEL},
  {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel, SITE_KERNEL},
  {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel_ptsz, SITE_KERNEL},
  {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchGridAsync, SITE_KERNEL},
  {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch, SITE_GRAPH},
  {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz, SITE_GRAPH},
  {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemcpy, SITE_MEMORY},
  {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemcpy_ptds, SITE_MEMORY},
  {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemcpyAsync, SITE_MEMORY},
  {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemcpyAsync_ptsz, SITE_MEMORY},
  {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD_v2, SITE_MEMORY},
  {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD_v2_ptds, SITE_MEMORY},
  {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2, SITE_MEMORY},
  {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2_ptsz, SITE_MEMORY},
  {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH_v2, SITE_MEMORY},
  {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH_v2_ptds, SITE_MEMORY},
  {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2, SITE_MEMORY},
  {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2_ptsz, SITE_MEMORY},
  {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoD_v2, SITE_MEMORY},
  {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoD_v2_ptds, SITE_MEMORY},
  {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2, SITE_MEMORY},
  {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2_ptsz, SITE_MEMORY},
  {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemsetD8_v2, SITE_MEMORY},
  {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemsetD32_v2, SITE_MEMORY},
  {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemsetD8Async, SITE_MEMORY},
  {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemsetD32Async, SITE_MEMORY},
  {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemAlloc_v2, SITE_ALLOC},
  {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemFree_v2, SITE_ALLOC},
  {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemAllocAsync, SITE_ALLOC},
  {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuMemFreeAsync, SITE_ALLOC},
};

static enum call_site_class call_site_class(CUpti_CallbackDomain domain, CUpti_CallbackId cbid)
{
  for (size_t i = 0; i < sizeof(g_call_sites) / sizeof(g_call_sites[0]); i++) {
    if (g_call_sites[i].domain == domain && g_call_sites[i].cbid == cbid) {
      return g_call_sites[i].cls;
    }
  }
  return SITE_NONE;
}

/* Correlation id of the tracked runtime API call last made on this thread. */
static __thread uint32_t t_runtime_corr;
static void CUPTIAPI
api_callback(void *userdata, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const void *cbinfo)
{
  (void)userdata;
  // NVTX delivers CUpti_NvtxData, not CUpti_CallbackData.
  if (domain == CUPTI_CB_DOMAIN_NVTX) {
    handle_nvtx((const CUpti_NvtxData *)cbinfo);
    return;
  }
  if (domain == CUPTI_CB_DOMAIN_RESOURCE) {
    if (cbid == CUPTI_CBID_RESOURCE_CONTEXT_CREATED) {
      uvm_configure_ctx(((const CUpti_ResourceData *)cbinfo)->context);
      pc_on_context_created(((const CUpti_ResourceData *)cbinfo)->context);
    }
    if (cbid == CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING) {
      pc_on_context_destroy(((const CUpti_ResourceData *)cbinfo)->context);
    }
    return;
  }
  const CUpti_CallbackData *cb = (const CUpti_CallbackData *)cbinfo;
  if (cb->callbackSite != CUPTI_API_ENTER) {
    return;
  }
  const enum call_site_class cls = call_site_class(domain, cbid);
  if (cls == SITE_NONE) {
    return;
  }
  // A runtime API call reaches the driver API with the same correlation id:
  // decide once, at the runtime call.
  if (domain == CUPTI_CB_DOMAIN_DRIVER_API && cb->correlationId == t_runtime_corr) {
    return;
  }
  if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
    t_runtime_corr = cb->correlationId;
  }

  // Sampled call sites fire on_launch, the host stack keyed by correlation id.
  // Without a profiler attached none is: its GPU records would lack the stack.
  const uint32_t weight = OTEL_GPU_ENABLED(ready) ? sample_call_site(cls) : 0;
  if (weight != 0) {
    const struct otel_gpu_launch_rec lrec = {
      .correlation_id = cb->correlationId,
      .sample_weight  = weight,
    };
    OTEL_GPU_PROBE(on_launch, &lrec);
  }
  remember_call_site(cb->correlationId, weight, weight != 0 ? t_nvtx_joined : NULL);
}

/* Unified-memory counters (migrations + page faults). Configuring before a
 * context exists fails, so this runs from CONTEXT_CREATED — once per DEVICE
 * (the config carries a deviceId; a single one-shot would cover only device
 * 0 in multi-GPU processes). */
static void uvm_configure_ctx(CUcontext ctx)
{
  uint32_t dev = 0;
  if (cuptiGetDeviceId(ctx, &dev) != CUPTI_SUCCESS || dev >= 64) {
    return;
  }
  static uint64_t done_mask;
  uint64_t bit = 1ull << dev;
  if (__atomic_fetch_or(&done_mask, bit, __ATOMIC_RELAXED) & bit) {
    return;
  }
  CUpti_ActivityUnifiedMemoryCounterConfig uvm[4]       = {0};
  const CUpti_ActivityUnifiedMemoryCounterKind kinds[4] = {
    CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD,
    CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH,
    CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT,
    CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT,
  };
  for (int i = 0; i < 4; i++) {
    uvm[i].scope    = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_SINGLE_DEVICE;
    uvm[i].kind     = kinds[i];
    uvm[i].deviceId = dev;
    uvm[i].enable   = 1;
  }
  CUptiResult r = cuptiActivityConfigureUnifiedMemoryCounter(uvm, 4);
  if (r == CUPTI_SUCCESS) {
    static int kind_enabled;
    if (!__atomic_exchange_n(&kind_enabled, 1, __ATOMIC_RELAXED)) {
      r = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER);
    }
  }
  if (r != CUPTI_SUCCESS) {
    emit_error((int)r, "UVM counter enable failed");
  }
}

/* ─── PC sampling: per-kernel stall reasons ──────────────────────────────────
 * Continuous PC sampling (Volta+, CUDA 11.3+): hardware samples warp program
 * counters with stall reasons per context; a worker thread drains them every
 * second into gpu_event records. Needs profiling permission
 * (RmProfilingAdminOnly=0 or root). Opt-in via OTEL_GPU_PC_SAMPLING=1. */

#define PC_MAX_CTX 16
#define PC_BUF_PCS 4000

static pthread_mutex_t g_pc_mu = PTHREAD_MUTEX_INITIALIZER;
static CUcontext g_pc_ctx[PC_MAX_CTX];
static int g_pc_nctx;
static CUcontext g_pc_pending[PC_MAX_CTX];
static int g_pc_npending;
static int g_pc_worker_started;
static int g_pc_failed;
static char **g_pc_stall_names; /* CUPTI-owned, stable */
static uint32_t *g_pc_stall_idx;
static size_t g_pc_nstall;
/* CUPTI fills g_pc_collect continuously; GetData drains into g_pc_data. */
static CUpti_PCSamplingData g_pc_collect;
static CUpti_PCSamplingData g_pc_data;

static int pc_fetch_stall_names(CUcontext ctx)
{
  if (g_pc_nstall > 0) {
    return 1;
  }
  CUpti_PCSamplingGetNumStallReasonsParams np = {
    .size = CUpti_PCSamplingGetNumStallReasonsParamsSize,
    .ctx  = ctx,
  };
  size_t n           = 0;
  np.numStallReasons = &n;
  CUptiResult r      = cuptiPCSamplingGetNumStallReasons(&np);
  if (r != CUPTI_SUCCESS || n == 0) {
    emit_error((int)r, "PC sampling: GetNumStallReasons failed");
    return 0;
  }
  g_pc_stall_names = (char **)calloc(n, sizeof(char *));
  g_pc_stall_idx   = (uint32_t *)calloc(n, sizeof(uint32_t));
  for (size_t i = 0; i < n; i++) {
    g_pc_stall_names[i] = (char *)calloc(128, 1);
  }
  CUpti_PCSamplingGetStallReasonsParams sp = {
    .size             = CUpti_PCSamplingGetStallReasonsParamsSize,
    .ctx              = ctx,
    .numStallReasons  = n,
    .stallReasonIndex = g_pc_stall_idx,
    .stallReasons     = g_pc_stall_names,
  };
  if (cuptiPCSamplingGetStallReasons(&sp) != CUPTI_SUCCESS) {
    return 0;
  }
  g_pc_nstall = n;
  return 1;
}

static const char *pc_stall_name(uint32_t idx)
{
  for (size_t i = 0; i < g_pc_nstall; i++) {
    if (g_pc_stall_idx[i] == idx) {
      return g_pc_stall_names[i];
    }
  }
  return NULL;
}

static void pc_drain_ctx(CUcontext ctx)
{
  CUpti_PCSamplingGetDataParams gp = {
    .size           = CUpti_PCSamplingGetDataParamsSize,
    .ctx            = ctx,
    .pcSamplingData = &g_pc_data,
  };
  if (cuptiPCSamplingGetData(&gp) != CUPTI_SUCCESS) {
    return;
  }
  for (size_t i = 0; i < g_pc_data.totalNumPcs && i < PC_BUF_PCS; i++) {
    CUpti_PCSamplingPCData *pc = &g_pc_data.pPcData[i];
    const char *fn             = nvtx_intern(pc->functionName); /* CUPTI reuses the buffer */
    if (!fn) {
      continue;
    }
    for (size_t s = 0; s < pc->stallReasonCount; s++) {
      const char *reason = pc_stall_name(pc->stallReason[s].pcSamplingStallReasonIndex);
      if (!reason || pc->stallReason[s].samples == 0) {
        continue;
      }
      struct otel_gpu_event_rec ev = {0};
      ev.kind                      = OTEL_GPU_EVENT_STALL;
      ev.name_ptr                  = (uint64_t)(uintptr_t)fn;
      ev.v1                        = (uint64_t)(uintptr_t)reason;
      ev.v2                        = pc->stallReason[s].samples;
      OTEL_GPU_PROBE(gpu_event, &ev);
    }
  }
}

/* PC sampling configuration may NOT run inside a CUPTI callback
 * (INVALID_OPERATION), so contexts are queued and set up on the worker. */
static void pc_setup_ctx(CUcontext ctx)
{
  /* Enable before stall-reason enumeration, per NVIDIA's
   * pc_sampling_continuous sample. Unverified end-to-end: T4 enumerates zero
   * stall reasons — don't rule out ordering when debugging that. */
  CUpti_PCSamplingEnableParams ep = {
    .size = CUpti_PCSamplingEnableParamsSize,
    .ctx  = ctx,
  };
  CUptiResult er = cuptiPCSamplingEnable(&ep);
  if (er != CUPTI_SUCCESS) {
    g_pc_failed = 1;
    emit_error((int)er, "PC sampling enable failed (needs RmProfilingAdminOnly=0 or root)");
    return;
  }
  if (!pc_fetch_stall_names(ctx)) {
    g_pc_failed = 1;
    return; // error emitted with the CUPTI result code by the fetch
  }
  if (g_pc_data.pPcData == NULL) {
    CUpti_PCSamplingData *bufs[2] = {&g_pc_collect, &g_pc_data};
    for (int b = 0; b < 2; b++) {
      bufs[b]->size          = sizeof(CUpti_PCSamplingData);
      bufs[b]->collectNumPcs = PC_BUF_PCS;
      bufs[b]->pPcData =
        (CUpti_PCSamplingPCData *)calloc(PC_BUF_PCS, sizeof(CUpti_PCSamplingPCData));
      for (size_t i = 0; i < PC_BUF_PCS; i++) {
        bufs[b]->pPcData[i].stallReason =
          (CUpti_PCSamplingStallReason *)calloc(g_pc_nstall, sizeof(CUpti_PCSamplingStallReason));
      }
    }
  }

  /* Continuous collection, period 2^7 cycles, all stall reasons. */
  CUpti_PCSamplingConfigurationInfo cfg[4] = {0};
  CUpti_PCSamplingCollectionMode mode      = CUPTI_PC_SAMPLING_COLLECTION_MODE_CONTINUOUS;
  cfg[0].attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_COLLECTION_MODE;
  cfg[0].attributeData.collectionModeData.collectionMode = mode;
  cfg[1].attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_STALL_REASON;
  cfg[1].attributeData.stallReasonData.stallReasonCount  = g_pc_nstall;
  cfg[1].attributeData.stallReasonData.pStallReasonIndex = g_pc_stall_idx;
  cfg[2].attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_DATA_BUFFER;
  cfg[2].attributeData.samplingDataBufferData.samplingDataBuffer = &g_pc_collect;
  cfg[3].attributeType = CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_PERIOD;
  cfg[3].attributeData.samplingPeriodData.samplingPeriod = 7;
  CUpti_PCSamplingConfigurationInfoParams cp             = {
                .size                         = CUpti_PCSamplingConfigurationInfoParamsSize,
                .ctx                          = ctx,
                .numAttributes                = 4,
                .pPCSamplingConfigurationInfo = cfg,
  };
  CUptiResult r = cuptiPCSamplingSetConfigurationAttribute(&cp);
  if (r != CUPTI_SUCCESS) {
    g_pc_failed = 1;
    emit_error((int)r, "PC sampling configuration failed");
    return;
  }
  if (g_pc_nctx < PC_MAX_CTX) {
    g_pc_ctx[g_pc_nctx++] = ctx;
  }
}

static void *pc_worker(void *arg)
{
  (void)arg;
  for (;;) {
    sleep(1);
    pthread_mutex_lock(&g_pc_mu);
    while (g_pc_npending > 0 && !g_pc_failed) {
      pc_setup_ctx(g_pc_pending[--g_pc_npending]);
    }
    for (int i = 0; i < g_pc_nctx; i++) {
      pc_drain_ctx(g_pc_ctx[i]);
    }
    pthread_mutex_unlock(&g_pc_mu);
  }
  return NULL;
}

/* The worker must never touch a dangling CUcontext. */
static void pc_on_context_destroy(CUcontext ctx)
{
  pthread_mutex_lock(&g_pc_mu);
  for (int i = 0; i < g_pc_npending; i++) {
    if (g_pc_pending[i] == ctx) {
      g_pc_pending[i] = g_pc_pending[--g_pc_npending];
      break;
    }
  }
  for (int i = 0; i < g_pc_nctx; i++) {
    if (g_pc_ctx[i] == ctx) {
      g_pc_ctx[i]                      = g_pc_ctx[--g_pc_nctx];
      CUpti_PCSamplingDisableParams dp = {
        .size = CUpti_PCSamplingDisableParamsSize,
        .ctx  = ctx,
      };
      cuptiPCSamplingDisable(&dp);
      break;
    }
  }
  pthread_mutex_unlock(&g_pc_mu);
}

static void pc_on_context_created(CUcontext ctx)
{
  /* Opt-in: hardware warp sampling inside customer workloads must cost
   * nothing by default. */
  if (g_pc_failed || !getenv("OTEL_GPU_PC_SAMPLING")) {
    return;
  }
  pthread_mutex_lock(&g_pc_mu);
  if (g_pc_npending < PC_MAX_CTX) {
    g_pc_pending[g_pc_npending++] = ctx;
  }
  if (!g_pc_worker_started) {
    g_pc_worker_started = 1;
    pthread_t t;
    pthread_attr_t a;
    pthread_attr_init(&a);
    pthread_attr_setdetachstate(&a, PTHREAD_CREATE_DETACHED);
    pthread_create(&t, &a, pc_worker, NULL);
    pthread_attr_destroy(&a);
  }
  pthread_mutex_unlock(&g_pc_mu);
}

static int setup(void)
{
  g_sample_rate[SITE_KERNEL] = env_rate("OTEL_GPU_SAMPLE_RATE", DEFAULT_SAMPLE_RATE);
  g_sample_rate[SITE_MEMORY] = g_sample_rate[SITE_KERNEL];
  g_sample_rate[SITE_ALLOC]  = g_sample_rate[SITE_KERNEL];
  /* Sampling switched off covers graph launches too, unless they are set apart. */
  g_sample_rate[SITE_GRAPH]  = env_rate(
    "OTEL_GPU_GRAPH_SAMPLE_RATE", g_sample_rate[SITE_KERNEL] == 0 ? 0 : DEFAULT_GRAPH_SAMPLE_RATE);

  CUptiResult r = cuptiSubscribe(&g_subscriber, (CUpti_CallbackFunc)api_callback, NULL);
  if (r == CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED) {
    emit_error(
      (int)r, "another CUPTI subscriber present (Kineto/Nsight/DCGM?); GPU profiling disabled");
    return 0;
  }
  if (r != CUPTI_SUCCESS) {
    emit_error((int)r, "cuptiSubscribe failed");
    return 0;
  }
  for (size_t i = 0; i < sizeof(g_call_sites) / sizeof(g_call_sites[0]); i++) {
    cuptiEnableCallback(1, g_subscriber, g_call_sites[i].domain, g_call_sites[i].cbid);
  }

  // Enabling the NVTX domain makes CUPTI register itself as the NVTX
  // injection, routing nvtxRangePush/Pop to api_callback.
  cuptiEnableDomain(1, g_subscriber, CUPTI_CB_DOMAIN_NVTX);
  cuptiEnableCallback(
    1, g_subscriber, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_CONTEXT_CREATED);
  cuptiEnableCallback(
    1, g_subscriber, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING);

  if (cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed) != CUPTI_SUCCESS) {
    emit_error(-1, "cuptiActivityRegisterCallbacks failed");
    return 0;
  }
  cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME);
  cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
  cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET);
  cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMORY2);

  /* Otherwise records sit until the 8 MiB buffer fills, delaying export and
   * aging out the call site table. */
  cuptiActivityFlushPeriod(1000);
  return 1;
}

static void flush_at_exit(void)
{
  if (g_enabled) {
    cuptiActivityFlushAll(1);
  }
  pthread_mutex_lock(&g_err_mu);
  if (g_err_started) {
    OTEL_GPU_PROBE(error, &g_last_err); /* last chance for the agent to see it */
  }
  pthread_mutex_unlock(&g_err_mu);
}

static void fork_prepare(void)
{
  pthread_mutex_lock(&g_err_mu);
  pthread_mutex_lock(&g_nvtx_pool_mu);
  pthread_mutex_lock(&g_pc_mu);
}
static void fork_parent(void)
{
  pthread_mutex_unlock(&g_pc_mu);
  pthread_mutex_unlock(&g_nvtx_pool_mu);
  pthread_mutex_unlock(&g_err_mu);
}

/* CUDA_INJECTION64_PATH entry point, called once at cuInit. Exported; the
 * rest of the lib is -fvisibility=hidden. */
__attribute__((visibility("default"))) int InitializeInjection(void)
{
  /* Hold our mutexes across fork: a child must never start with one locked
   * by a dead thread (PyTorch DataLoader forks while threads push NVTX). */
  pthread_atfork(fork_prepare, fork_parent, fork_parent);
  OTEL_GPU_PROBE(ready, NULL);
  g_enabled = setup();
  atexit(flush_at_exit); /* error re-emit starts from emit_error itself */
  return 1;              /* must return non-zero; lib stays loaded regardless of our state */
}
