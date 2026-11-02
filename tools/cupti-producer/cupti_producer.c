// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0
//
// libotelcupti.so — CUDA injection library (CUDA_INJECTION64_PATH), a
// reference producer of the otel_gpu_v1 USDT probe consumed by probes/gpu.
// Captures GPU calls and timing through CUPTI and sends them as events of the
// otel_gpu_v1:event probe (probes/gpu/otel_gpu_v1_usdt.h).
//
// Plain C on purpose: the library loads into arbitrary host processes, so no
// libstdc++/C++ runtime ABI exposure (demangling happens on the Go side).
//
// Coexistence: CUPTI allows one subscriber per process. If another tool
// subscribed first, we log an error to stderr and disable gracefully.

#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <cuda.h> /* CUDA_VERSION */
#include <cupti.h>
#include <nvtx3/nvToolsExt.h> /* nvtxEventAttributes_t for NVTX range messages (NVTX3) */

#if CUDA_VERSION < 12000
  #error "CUDA 12 or newer is required"
#endif

#define OTEL_GPU_SEMAPHORE
#include "cuda_driver.h"
#include "otel_gpu_v1_usdt.h"
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

static void log_error(int code, const char *msg)
{
  fprintf(stderr, "otelcupti: %s (%d)\n", msg, code);
}

/* ─── NVTX range tracking ────────────────────────────────────────────────────
 * NVTX ranges carry app-level names stripped host stacks lack. Keep a
 * per-thread range stack, intern the names, and keep the active nesting per
 * call so buffer_completed can attach it per kernel. */
#define NVTX_MAX_DEPTH   64
#define NVTX_POOL_MAX    (1 << 16)
/* Slots looked at for a string: a string not found within them is not kept, so
 * a full pool costs a few comparisons rather than a scan of all of it. */
#define NVTX_POOL_PROBES 64

/* Intern: dedup + strdup, NEVER freed — the call table keeps pointers to the
 * joined stacks until the work of a call is reported, from another thread.
 * Insert-only open-addressed hash: lock-free reads, inserts take the mutex and
 * re-probe. Holds NVTX names and their joined stacks. */
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
  for (uint32_t i = 0; i < NVTX_POOL_PROBES; i++) {
    uint32_t slot = (h + i) & (NVTX_POOL_MAX - 1);
    const char *p = __atomic_load_n(&g_nvtx_pool[slot], __ATOMIC_ACQUIRE);
    if (p == NULL) {
      pthread_mutex_lock(&g_nvtx_pool_mu);
      /* Re-probe under the lock: another thread may have inserted. */
      const char *found = NULL;
      for (; i < NVTX_POOL_PROBES; i++) {
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
      return found; /* NULL when the slots filled up concurrently */
    }
    if (strcmp(p, s) == 0) {
      return p;
    }
  }
  return NULL; /* the slots of this string are full */
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

/* ─── Calls by correlation id ────────────────────────────────────────────────
 * Every tracked call leaves its sample weight and NVTX nesting here, for
 * buffer_completed to filter, weigh and label its GPU records. Direct-mapped
 * on sequential correlation ids = FIFO of the last SLOTS ids. CUDA gives every
 * API call an id, tracked or not, so this must exceed the calls made between
 * a call and the flush of its records (FLUSH_PERIOD_MS, plus the time its work
 * waits on the GPU). Zero-page BSS. */
#define CALL_SLOTS      (1 << 20) /* power of two */
#define FLUSH_PERIOD_MS 500

/* correlation id << 32 | sample weight, 0 if not sampled; packed so a reader
 * sees both or neither. A torn read of the NVTX nesting only mislabels a
 * kernel. */
static uint64_t g_call_key[CALL_SLOTS];
static const char *g_call_nvtx[CALL_SLOTS];

/* CUPTI timestamp at which tracing started, UINT64_MAX while it is off: the GPU
 * work of an untracked call that started earlier may come from before the
 * profiler attached. */
static uint64_t g_tracing_since = UINT64_MAX;
/* Whether CUPTI records GPU work: LAUNCH events are sent only then. */
static int g_tracing;

static void remember_call(uint32_t corr, uint32_t weight, const char *nvtx)
{
  if (corr == 0) {
    return;
  }
  uint32_t h = corr & (CALL_SLOTS - 1);
  __atomic_store_n(&g_call_nvtx[h], nvtx, __ATOMIC_RELEASE);
  __atomic_store_n(&g_call_key[h], (uint64_t)corr << 32 | weight, __ATOMIC_RELEASE);
}

static void report_lost_calls(void)
{
  static int reported;
  if (!__atomic_exchange_n(&reported, 1, __ATOMIC_RELAXED)) {
    log_error(-1, "call table too small: GPU records were dropped");
  }
}

/* Returns the sample weight of the call corr, 0 if its GPU records are not
 * reported, and its NVTX nesting. The records of an untracked call, which sent
 * no LAUNCH event, count once if they started after tracing did. */
static uint32_t call_weight(uint32_t corr, uint64_t start, const char **nvtx)
{
  *nvtx = NULL;
  if (corr != 0) {
    uint32_t h          = corr & (CALL_SLOTS - 1);
    const uint64_t key  = __atomic_load_n(&g_call_key[h], __ATOMIC_ACQUIRE);
    const uint32_t call = (uint32_t)(key >> 32);
    if (key != 0 && call == corr) {
      *nvtx = __atomic_load_n(&g_call_nvtx[h], __ATOMIC_ACQUIRE);
      return (uint32_t)key;
    }
    if (key != 0 && (int32_t)(call - corr) > 0) {
      /* A later call took the slot: whether this one was sampled is lost. */
      report_lost_calls();
      return 0;
    }
  }
  return start >= __atomic_load_n(&g_tracing_since, __ATOMIC_ACQUIRE) ? 1 : 0;
}

/* ─── Call sampling ──────────────────────────────────────────────────────────
 * Kernel launches, graph launches and memory operations are sampled apart, so
 * a rare class keeps its samples next to a frequent one. */
enum call_class {
  CALL_UNTRACKED,
  CALL_KERNEL,
  CALL_GRAPH,
  CALL_MEMORY,
  CALL_CLASSES,
};

/* Default sampled calls per thread per second. Graph launches are sampled
 * less, as each runs many kernels. */
#define DEFAULT_SAMPLE_RATE       200
#define DEFAULT_GRAPH_SAMPLE_RATE 20

/* Sampled calls per thread per second, by class; 0 samples all of them. */
static uint32_t g_sample_rate[CALL_CLASSES];

static __thread struct otel_sampler t_samplers[CALL_CLASSES];
static __thread uint64_t t_rng;

static uint64_t now_ns(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static uint32_t sample_call(enum call_class cls)
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

/* ─── Debug totals ───────────────────────────────────────────────────────────
 * With OTEL_GPU_DEBUG_TOTALS set, the producer adds up the GPU work that
 * started while it traced, sampled or not, and what its samples stand for. It
 * prints both every DEBUG_TOTALS_PERIOD_MS while tracing, and when tracing
 * stops or the process exits: a profile of the same period adds up to the
 * second, an estimate of the first. */
#define DEBUG_TOTALS_PERIOD_MS 10000

struct work_totals {
  uint64_t kernel_ns;
  uint64_t memory_ns;
  uint64_t memory_bytes;
};

static int g_debug_totals;
static struct work_totals g_exact_totals;
static struct work_totals g_estimated_totals;
static uint32_t g_largest_payload;

static void
add_totals(struct work_totals *t, int memory, uint64_t ns, uint64_t bytes, uint32_t times)
{
  __atomic_fetch_add(memory ? &t->memory_ns : &t->kernel_ns, ns * times, __ATOMIC_RELAXED);
  __atomic_fetch_add(&t->memory_bytes, bytes * times, __ATOMIC_RELAXED);
}

static void count_work(int memory, uint64_t start, uint64_t end, uint64_t bytes, uint32_t weight)
{
  if (!g_debug_totals || start < __atomic_load_n(&g_tracing_since, __ATOMIC_ACQUIRE)) {
    return;
  }
  const uint64_t ns = end > start ? end - start : 0;
  add_totals(&g_exact_totals, memory, ns, bytes, 1);
  add_totals(&g_estimated_totals, memory, ns, bytes, weight);
}

static void count_payload(const struct otel_gpu_event *ev)
{
  uint32_t largest = __atomic_load_n(&g_largest_payload, __ATOMIC_RELAXED);
  while (ev->payload_size > largest &&
         !__atomic_compare_exchange_n(
           &g_largest_payload, &largest, ev->payload_size, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
  }
}

static unsigned long long read_total(uint64_t *v, int reset)
{
  return reset ? __atomic_exchange_n(v, 0, __ATOMIC_RELAXED) : __atomic_load_n(v, __ATOMIC_RELAXED);
}

/* Prints the totals since tracing started; reset starts new ones. */
static void print_totals(int reset)
{
  fprintf(
    stderr,
    "otelcupti: debug totals: pid=%d exact_kernel_ns=%llu exact_memory_ns=%llu "
    "exact_memory_bytes=%llu estimated_kernel_ns=%llu estimated_memory_ns=%llu "
    "estimated_memory_bytes=%llu largest_payload=%u\n",
    (int)getpid(),
    read_total(&g_exact_totals.kernel_ns, reset),
    read_total(&g_exact_totals.memory_ns, reset),
    read_total(&g_exact_totals.memory_bytes, reset),
    read_total(&g_estimated_totals.kernel_ns, reset),
    read_total(&g_estimated_totals.memory_ns, reset),
    read_total(&g_estimated_totals.memory_bytes, reset),
    reset ? __atomic_exchange_n(&g_largest_payload, 0, __ATOMIC_RELAXED)
          : __atomic_load_n(&g_largest_payload, __ATOMIC_RELAXED));
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

/* emit_event is the only place firing the probe, so the profiler attaches one
 * uprobe to the library. */
__attribute__((noinline)) static void emit_event(const struct otel_gpu_event *ev)
{
  OTEL_GPU_EVENT(ev);
}

static void
begin_event(struct otel_gpu_event *ev, enum otel_gpu_kind kind, uint32_t corr, uint32_t weight)
{
  ev->correlation_id = corr;
  ev->sample_weight  = weight;
  ev->kind           = (uint16_t)kind;
  ev->payload_size   = 0;
}

/* put_work adds the fields of an event of GPU work: the GPU, how long it took,
 * how many bytes it moved, its name, and the names of the work in progress at
 * its call, innermost first. Names come last: one too long for the payload is
 * cut short, and must not crowd out the others. */
static void put_work(
  struct otel_gpu_event *ev,
  uint32_t device,
  uint64_t start,
  uint64_t end,
  uint64_t bytes,
  const char *name,
  const char *nvtx)
{
  otel_gpu_put_string(ev, OTEL_GPU_FIELD_GPU, gpu_uuid(device));
  otel_gpu_put_uint(ev, OTEL_GPU_FIELD_DURATION_NS, end > start ? end - start : 0);
  otel_gpu_put_uint(ev, OTEL_GPU_FIELD_BYTES, bytes);
  otel_gpu_put_string(ev, OTEL_GPU_FIELD_NAME, name);
  for (const char *range = nvtx; range && *range;) {
    const char *sep  = strchr(range, NVTX_SEP);
    const size_t len = sep ? (size_t)(sep - range) : strlen(range);
    otel_gpu_put_bytes(ev, OTEL_GPU_FIELD_RANGES, range, len);
    range = sep ? sep + 1 : NULL;
  }
  if (g_debug_totals) {
    count_payload(ev);
  }
}

static void report_kernel(const otelcupti_kernel_activity_t *k)
{
  const char *nvtx;
  const uint32_t weight = call_weight(k->correlationId, k->start, &nvtx);
  count_work(0, k->start, k->end, 0, weight);
  if (weight == 0) {
    return;
  }
  struct otel_gpu_event ev;
  begin_event(&ev, OTEL_GPU_KERNEL, k->correlationId, weight);
  put_work(&ev, k->deviceId, k->start, k->end, 0, k->name, nvtx);
  emit_event(&ev);
}

/* Names of CUpti_ActivityMemcpyKind values: where the data went. */
static const char *const g_copy_kinds[] = {
  [CUPTI_ACTIVITY_MEMCPY_KIND_HTOD] = "HtoD",
  [CUPTI_ACTIVITY_MEMCPY_KIND_DTOH] = "DtoH",
  [CUPTI_ACTIVITY_MEMCPY_KIND_HTOA] = "HtoA",
  [CUPTI_ACTIVITY_MEMCPY_KIND_ATOH] = "AtoH",
  [CUPTI_ACTIVITY_MEMCPY_KIND_ATOA] = "AtoA",
  [CUPTI_ACTIVITY_MEMCPY_KIND_ATOD] = "AtoD",
  [CUPTI_ACTIVITY_MEMCPY_KIND_DTOA] = "DtoA",
  [CUPTI_ACTIVITY_MEMCPY_KIND_DTOD] = "DtoD",
  [CUPTI_ACTIVITY_MEMCPY_KIND_HTOH] = "HtoH",
};

static void report_memcpy(const CUpti_ActivityMemcpy *m)
{
  const char *nvtx;
  const uint32_t weight = call_weight(m->correlationId, m->start, &nvtx);
  count_work(1, m->start, m->end, m->bytes, weight);
  if (weight == 0) {
    return;
  }
  const char *name =
    m->copyKind < sizeof(g_copy_kinds) / sizeof(g_copy_kinds[0]) ? g_copy_kinds[m->copyKind] : NULL;
  struct otel_gpu_event ev;
  begin_event(&ev, OTEL_GPU_COPY, m->correlationId, weight);
  put_work(&ev, m->deviceId, m->start, m->end, m->bytes, name, nvtx);
  emit_event(&ev);
}

static void report_memset(const CUpti_ActivityMemset *m)
{
  const char *nvtx;
  const uint32_t weight = call_weight(m->correlationId, m->start, &nvtx);
  count_work(1, m->start, m->end, m->bytes, weight);
  if (weight == 0) {
    return;
  }
  struct otel_gpu_event ev;
  begin_event(&ev, OTEL_GPU_FILL, m->correlationId, weight);
  put_work(&ev, m->deviceId, m->start, m->end, m->bytes, NULL, nvtx);
  emit_event(&ev);
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
        report_kernel((const otelcupti_kernel_activity_t *)record);
        break;
      case CUPTI_ACTIVITY_KIND_MEMCPY: report_memcpy((const CUpti_ActivityMemcpy *)record); break;
      case CUPTI_ACTIVITY_KIND_MEMSET: report_memset((const CUpti_ActivityMemset *)record); break;
      default: break;
      }
    } else if (st != CUPTI_ERROR_MAX_LIMIT_REACHED) {
      break;
    }
  } while (st == CUPTI_SUCCESS);

  free(buffer); /* posix_memalign'd in buffer_requested → same pointer */
}

/* ─── Callback API: launch correlation (CPU side) ──────────────────────────── */

/* Tracked calls send LAUNCH events: the runtime and driver API calls that
 * launch kernels or graphs, or copy or fill memory. They are found by name at
 * start, so that every variant is tracked, also those of later CUDA versions.
 * The first matching prefix decides. */
static const struct {
  const char *prefix;
  enum call_class cls;
} g_call_prefixes[] = {
  /* Host functions run on the CPU. */
  {"cudaLaunchHostFunc", CALL_UNTRACKED},
  {"cuLaunchHostFunc", CALL_UNTRACKED},
  {"cudaGraphLaunch", CALL_GRAPH},
  {"cuGraphLaunch", CALL_GRAPH},
  {"cudaLaunch", CALL_KERNEL},
  {"cuLaunch", CALL_KERNEL},
  {"cudaMemcpy", CALL_MEMORY},
  {"cudaMemset", CALL_MEMORY},
  {"cuMemcpy", CALL_MEMORY},
  {"cuMemset", CALL_MEMORY},
};

/* The class of each runtime and driver API call, filled at start. */
static uint8_t g_runtime_class[CUPTI_RUNTIME_TRACE_CBID_SIZE];
static uint8_t g_driver_class[CUPTI_DRIVER_TRACE_CBID_SIZE];

static enum call_class classify_call(const char *name)
{
  for (size_t i = 0; i < sizeof(g_call_prefixes) / sizeof(g_call_prefixes[0]); i++) {
    const char *prefix = g_call_prefixes[i].prefix;
    if (strncmp(name, prefix, strlen(prefix)) == 0) {
      return g_call_prefixes[i].cls;
    }
  }
  return CALL_UNTRACKED;
}

/* Enables the callbacks of the tracked calls among the count calls of a domain,
 * and returns how many could not be enabled. */
static uint32_t track_calls(CUpti_CallbackDomain domain, uint8_t *classes, uint32_t count)
{
  uint32_t failed = 0;
  for (uint32_t cbid = 0; cbid < count; cbid++) {
    const char *name = NULL;
    if (cuptiGetCallbackName(domain, cbid, &name) != CUPTI_SUCCESS || !name) {
      continue;
    }
    const enum call_class cls = classify_call(name);
    if (cls == CALL_UNTRACKED) {
      continue;
    }
    if (cuptiEnableCallback(1, g_subscriber, domain, cbid) != CUPTI_SUCCESS) {
      failed++;
      continue;
    }
    classes[cbid] = (uint8_t)cls;
  }
  return failed;
}

static enum call_class call_class(CUpti_CallbackDomain domain, CUpti_CallbackId cbid)
{
  if (domain == CUPTI_CB_DOMAIN_RUNTIME_API && cbid < CUPTI_RUNTIME_TRACE_CBID_SIZE) {
    return (enum call_class)g_runtime_class[cbid];
  }
  if (domain == CUPTI_CB_DOMAIN_DRIVER_API && cbid < CUPTI_DRIVER_TRACE_CBID_SIZE) {
    return (enum call_class)g_driver_class[cbid];
  }
  return CALL_UNTRACKED;
}

/* Correlation id of the call last decided on this thread: a runtime call and
 * the driver calls it makes share it, and are decided once. */
static __thread uint32_t t_decided_corr;
static void CUPTIAPI
api_callback(void *userdata, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const void *cbinfo)
{
  (void)userdata;
  // NVTX delivers CUpti_NvtxData, not CUpti_CallbackData.
  if (domain == CUPTI_CB_DOMAIN_NVTX) {
    handle_nvtx((const CUpti_NvtxData *)cbinfo);
    return;
  }
  const CUpti_CallbackData *cb = (const CUpti_CallbackData *)cbinfo;
  if (cb->callbackSite != CUPTI_API_ENTER) {
    return;
  }
  const enum call_class cls = call_class(domain, cbid);
  if (cls == CALL_UNTRACKED) {
    return;
  }
  // A runtime API call reaches the driver API with the same correlation id,
  // and so may several driver calls of one untracked runtime call.
  if (cb->correlationId == t_decided_corr) {
    return;
  }
  t_decided_corr = cb->correlationId;

  // Sampled calls send a LAUNCH event, for the profiler to record the host
  // stack under the correlation id. None is sent while CUPTI does not record
  // GPU work: the call's GPU records would never come.
  const int reporting   = OTEL_GPU_ENABLED() && __atomic_load_n(&g_tracing, __ATOMIC_ACQUIRE);
  const uint32_t weight = reporting ? sample_call(cls) : 0;
  if (weight != 0) {
    struct otel_gpu_event ev;
    begin_event(&ev, OTEL_GPU_LAUNCH, cb->correlationId, weight);
    emit_event(&ev);
  }
  remember_call(cb->correlationId, weight, weight != 0 ? t_nvtx_joined : NULL);
}

/* ─── Tracing while the profiler is attached ────────────────────────────────
 * The activity kinds are on only while the probe's semaphore is raised, so the
 * producer costs little without a profiler: a check of the semaphore at each
 * tracked call. */
static pthread_mutex_t g_tracing_mu = PTHREAD_MUTEX_INITIALIZER;
/* Set once tracing could not start: the producer then reports nothing. */
static int g_tracing_failed;

static const CUpti_ActivityKind g_activity_kinds[] = {
  CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
  CUPTI_ACTIVITY_KIND_MEMCPY,
  CUPTI_ACTIVITY_KIND_MEMSET,
};
#define NUM_ACTIVITY_KINDS (sizeof(g_activity_kinds) / sizeof(g_activity_kinds[0]))

static void disable_activity(size_t count)
{
  for (size_t i = 0; i < count; i++) {
    CUptiResult r = cuptiActivityDisable(g_activity_kinds[i]);
    if (r != CUPTI_SUCCESS) {
      log_error((int)r, "activity disable failed");
    }
  }
}

/* Starts tracing. Called with g_tracing_mu held. */
static void start_tracing(void)
{
  uint64_t since = 0;
  cuptiGetTimestamp(&since);
  __atomic_store_n(&g_tracing_since, since, __ATOMIC_RELEASE);
  for (size_t i = 0; i < NUM_ACTIVITY_KINDS; i++) {
    CUptiResult r = cuptiActivityEnable(g_activity_kinds[i]);
    if (r != CUPTI_SUCCESS) {
      log_error((int)r, "activity enable failed; GPU profiling disabled");
      disable_activity(i);
      __atomic_store_n(&g_tracing_since, UINT64_MAX, __ATOMIC_RELEASE);
      g_tracing_failed = 1;
      return;
    }
  }
  __atomic_store_n(&g_tracing, 1, __ATOMIC_RELEASE);
}

/* Stops tracing. Called with g_tracing_mu held. */
static void stop_tracing(void)
{
  __atomic_store_n(&g_tracing, 0, __ATOMIC_RELEASE);
  disable_activity(NUM_ACTIVITY_KINDS);
  if (g_debug_totals) {
    cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
    print_totals(1);
  }
  __atomic_store_n(&g_tracing_since, UINT64_MAX, __ATOMIC_RELEASE);
}

/* Follows the probe's semaphore: on when the profiler attached, off when it
 * left. */
static void update_tracing(void)
{
  const int want = OTEL_GPU_ENABLED() != 0;
  if (want == __atomic_load_n(&g_tracing, __ATOMIC_ACQUIRE)) {
    return;
  }
  pthread_mutex_lock(&g_tracing_mu);
  if (want != g_tracing && !g_tracing_failed) {
    if (want) {
      start_tracing();
    } else {
      stop_tracing();
    }
  }
  pthread_mutex_unlock(&g_tracing_mu);
}

static void start_detached(void *(*fn)(void *))
{
  pthread_t t;
  pthread_attr_t a;
  pthread_attr_init(&a);
  pthread_attr_setdetachstate(&a, PTHREAD_CREATE_DETACHED);
  pthread_create(&t, &a, fn, NULL);
  pthread_attr_destroy(&a);
}

static void *tracing_worker(void *arg)
{
  (void)arg;
  for (uint64_t round = 1;; round++) {
    // Here, not in the application's CUDA calls: turning tracing off can take
    // seconds.
    update_tracing();
    if (__atomic_load_n(&g_tracing, __ATOMIC_ACQUIRE)) {
      // CUPTI hands over a buffer of records only once it is full, unless asked.
      cuptiActivityFlushAll(0);
      if (g_debug_totals && round % (DEBUG_TOTALS_PERIOD_MS / FLUSH_PERIOD_MS) == 0) {
        print_totals(0);
      }
    }
    usleep(FLUSH_PERIOD_MS * 1000);
  }
  return NULL;
}

static int setup(void)
{
  g_sample_rate[CALL_KERNEL] = env_rate("OTEL_GPU_SAMPLE_RATE", DEFAULT_SAMPLE_RATE);
  g_sample_rate[CALL_MEMORY] = g_sample_rate[CALL_KERNEL];
  /* Sampling switched off covers graph launches too, unless they are set apart. */
  g_sample_rate[CALL_GRAPH]  = env_rate(
    "OTEL_GPU_GRAPH_SAMPLE_RATE", g_sample_rate[CALL_KERNEL] == 0 ? 0 : DEFAULT_GRAPH_SAMPLE_RATE);
  const char *totals = getenv("OTEL_GPU_DEBUG_TOTALS");
  g_debug_totals     = totals && *totals && strcmp(totals, "0") != 0;

  CUptiResult r = cuptiSubscribe(&g_subscriber, (CUpti_CallbackFunc)api_callback, NULL);
  if (r == CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED) {
    log_error((int)r, "another CUPTI subscriber present; GPU profiling disabled");
    return 0;
  }
  if (r != CUPTI_SUCCESS) {
    log_error((int)r, "cuptiSubscribe failed");
    return 0;
  }
  const uint32_t untracked =
    track_calls(CUPTI_CB_DOMAIN_RUNTIME_API, g_runtime_class, CUPTI_RUNTIME_TRACE_CBID_SIZE) +
    track_calls(CUPTI_CB_DOMAIN_DRIVER_API, g_driver_class, CUPTI_DRIVER_TRACE_CBID_SIZE);
  if (untracked != 0) {
    log_error((int)untracked, "GPU calls not tracked, their work is reported without a stack");
  }

  // Enabling the NVTX domain routes to api_callback the ranges of an NVTX
  // library that NVTX_INJECTION64_PATH points at this CUPTI.
  r = cuptiEnableDomain(1, g_subscriber, CUPTI_CB_DOMAIN_NVTX);
  if (r != CUPTI_SUCCESS) {
    log_error((int)r, "NVTX ranges unavailable");
  }

  r = cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed);
  if (r != CUPTI_SUCCESS) {
    log_error((int)r, "cuptiActivityRegisterCallbacks failed");
    cuptiUnsubscribe(g_subscriber);
    return 0;
  }

  update_tracing();
  start_detached(tracing_worker);
  return 1;
}

static void flush_at_exit(void)
{
  if (!g_enabled) {
    return;
  }
  cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
  if (g_debug_totals && __atomic_load_n(&g_tracing, __ATOMIC_ACQUIRE)) {
    print_totals(0);
  }
}

/* Locks taken in the order the producer's threads nest them, so that fork
 * never waits on a lock held by a thread that waits on another. */
static void fork_prepare(void)
{
  pthread_mutex_lock(&g_tracing_mu);
  pthread_mutex_lock(&g_nvtx_pool_mu);
  pthread_mutex_lock(&g_gpu_mu);
}
static void fork_parent(void)
{
  pthread_mutex_unlock(&g_gpu_mu);
  pthread_mutex_unlock(&g_nvtx_pool_mu);
  pthread_mutex_unlock(&g_tracing_mu);
}
/* A child of a CUDA process cannot use CUDA, nor flush CUPTI at its exit. */
static void fork_child(void)
{
  fork_parent();
  g_enabled = 0;
}

/* CUDA_INJECTION64_PATH entry point, called once at cuInit. Exported; the
 * rest of the lib is -fvisibility=hidden. */
__attribute__((visibility("default"))) int InitializeInjection(void)
{
  /* Hold our mutexes across fork: a child must never start with one locked
   * by a dead thread (PyTorch DataLoader forks while threads push NVTX). */
  pthread_atfork(fork_prepare, fork_parent, fork_child);
  g_enabled = setup();
  atexit(flush_at_exit);
  return 1; /* must return non-zero; lib stays loaded regardless of our state */
}
