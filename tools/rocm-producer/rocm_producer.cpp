// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0
//
// libotelrocm.so — rocprofiler-sdk tool library (ROCP_TOOL_LIBRARIES), a
// reference producer of the otel_gpu_v1 USDT probe for AMD GPUs, the ROCm
// counterpart of tools/cupti-producer. The contract is
// probes/gpu/otel_gpu_v1_usdt.h.
//
// C++ only because the rocprofiler-sdk headers are: the code avoids the C++
// standard library, so the library does not depend on libstdc++ (it loads
// into arbitrary processes, where a second C++ runtime can clash).

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#define OTEL_GPU_SEMAPHORE
#include "otel_gpu_v1_usdt.h"
#include "sampler.h"

#define BUFFER_SIZE      (8u << 20)
#define BUFFER_WATERMARK (BUFFER_SIZE / 2)

/* g_context follows HIP calls, ranges and code objects all along;
 * g_trace_context records GPU work only while the profiler is attached. */
static rocprofiler_context_id_t g_context;
static rocprofiler_context_id_t g_trace_context;
static rocprofiler_buffer_id_t g_buffer;
static int g_enabled;

static void log_error(int code, const char *msg)
{
  fprintf(stderr, "otelrocm: %s (%d)\n", msg, code);
}

static void start_thread(void *(*fn)(void *))
{
  pthread_t t;
  pthread_attr_t a;
  pthread_attr_init(&a);
  pthread_attr_setdetachstate(&a, PTHREAD_CREATE_DETACHED);
  pthread_create(&t, &a, fn, NULL);
  pthread_attr_destroy(&a);
}

/* ─── Interned strings ──────────────────────────────────────────────────────
 * Kernel names and range stacks are looked up when the work of a call is
 * reported, from another thread, so each is copied here once and never freed.
 * Insert-only open-addressed table: lock-free reads, inserts take the mutex. */
#define STRING_POOL_SLOTS  (1 << 16)
/* Slots looked at for a string: a string not found within them is not kept, so
 * a full pool costs a few comparisons rather than a scan of all of it. */
#define STRING_POOL_PROBES 64

static pthread_mutex_t g_pool_mu = PTHREAD_MUTEX_INITIALIZER;
static const char *g_pool[STRING_POOL_SLOTS];

static uint32_t fnv1a(const char *s)
{
  uint32_t h = 2166136261u;
  for (; *s; s++) {
    h = (h ^ (uint8_t)*s) * 16777619u;
  }
  return h;
}

static const char *intern(const char *s)
{
  if (!s || !*s) {
    return NULL;
  }
  const uint32_t h = fnv1a(s);
  for (uint32_t i = 0; i < STRING_POOL_PROBES; i++) {
    uint32_t slot = (h + i) & (STRING_POOL_SLOTS - 1);
    const char *p = __atomic_load_n(&g_pool[slot], __ATOMIC_ACQUIRE);
    if (p == NULL) {
      pthread_mutex_lock(&g_pool_mu);
      /* Re-probe under the lock: another thread may have inserted. */
      const char *found = NULL;
      for (; i < STRING_POOL_PROBES; i++) {
        slot = (h + i) & (STRING_POOL_SLOTS - 1);
        p    = g_pool[slot];
        if (p == NULL) {
          found = strdup(s);
          __atomic_store_n(&g_pool[slot], found, __ATOMIC_RELEASE);
          break;
        }
        if (strcmp(p, s) == 0) {
          found = p;
          break;
        }
      }
      pthread_mutex_unlock(&g_pool_mu);
      return found;
    }
    if (strcmp(p, s) == 0) {
      return p;
    }
  }
  return NULL;
}

/* ─── Kernel names ──────────────────────────────────────────────────────────
 * Dispatch records carry a kernel id; code object callbacks register its name.
 * Insert-only, keyed by kernel id. */
#define KERNEL_SLOTS (1 << 16)

static uint64_t g_kernel_id[KERNEL_SLOTS];
static const char *g_kernel_name[KERNEL_SLOTS];

static void register_kernel(uint64_t id, const char *name)
{
  const char *interned = intern(name);
  pthread_mutex_lock(&g_pool_mu);
  for (uint32_t i = 0; i < KERNEL_SLOTS; i++) {
    const uint32_t slot = (uint32_t)(id + i) & (KERNEL_SLOTS - 1);
    if (g_kernel_id[slot] == 0 || g_kernel_id[slot] == id + 1) {
      __atomic_store_n(&g_kernel_name[slot], interned, __ATOMIC_RELEASE);
      __atomic_store_n(&g_kernel_id[slot], id + 1, __ATOMIC_RELEASE);
      break;
    }
  }
  pthread_mutex_unlock(&g_pool_mu);
}

static const char *kernel_name(uint64_t id)
{
  for (uint32_t i = 0; i < KERNEL_SLOTS; i++) {
    const uint32_t slot = (uint32_t)(id + i) & (KERNEL_SLOTS - 1);
    const uint64_t key  = __atomic_load_n(&g_kernel_id[slot], __ATOMIC_ACQUIRE);
    if (key == id + 1) {
      return __atomic_load_n(&g_kernel_name[slot], __ATOMIC_ACQUIRE);
    }
    if (key == 0) {
      break;
    }
  }
  return NULL;
}

/* ─── GPUs ──────────────────────────────────────────────────────────────────
 * Records name agents by opaque handle; the contract wants the GPU's UUID. */
#define MAX_AGENTS 64

static uint64_t g_agent_handle[MAX_AGENTS];
/* UUID of each GPU agent, 8-4-4-4-12 hex digits; empty for other agents */
static char g_agent_uuid[MAX_AGENTS][sizeof("00000000-0000-0000-0000-000000000000")];
static size_t g_num_agents;

static void format_uuid(char *out, const rocprofiler_uuid_t *uuid)
{
  for (size_t i = 0; i < sizeof(uuid->bytes); i++) {
    if (i == 4 || i == 6 || i == 8 || i == 10) {
      *out++ = '-';
    }
    out += sprintf(out, "%02x", uuid->bytes[i]);
  }
}

static rocprofiler_status_t
collect_agents(rocprofiler_agent_version_t, const void **agents, size_t num_agents, void *)
{
  for (size_t i = 0; i < num_agents && g_num_agents < MAX_AGENTS; i++) {
    const rocprofiler_agent_v0_t *a = (const rocprofiler_agent_v0_t *)agents[i];
    g_agent_handle[g_num_agents]    = a->id.handle;
    if (a->type == ROCPROFILER_AGENT_TYPE_GPU) {
      format_uuid(g_agent_uuid[g_num_agents], &a->uuid);
    }
    g_num_agents++;
  }
  return ROCPROFILER_STATUS_SUCCESS;
}

/* ─── Range stacks ──────────────────────────────────────────────────────────
 * ROCTx ranges (roctxRangePush/Pop), the counterpart of NVTX ranges. Each
 * thread keeps its active ranges as one interned string, innermost first and
 * separated by 0x1f, recomputed at push and pop. */
#define RANGE_MAX_DEPTH 64
#define RANGE_SEP       '\x1f'

static __thread const char *t_range_stack[RANGE_MAX_DEPTH];
static __thread int t_range_depth;
static __thread const char *t_ranges;

static void ranges_recompute(void)
{
  t_ranges    = NULL;
  const int n = t_range_depth < RANGE_MAX_DEPTH ? t_range_depth : RANGE_MAX_DEPTH;
  static __thread char buf[1024];
  size_t off = 0;
  for (int i = n - 1; i >= 0; i--) {
    const char *s = t_range_stack[i];
    if (!s) {
      continue;
    }
    const size_t l = strlen(s);
    if (off + l + 2 >= sizeof(buf)) {
      break;
    }
    if (off) {
      buf[off++] = RANGE_SEP;
    }
    memcpy(buf + off, s, l);
    off += l;
  }
  if (off > 0) {
    buf[off] = 0;
    t_ranges = intern(buf);
  }
}

static void range_push(const char *message)
{
  if (t_range_depth < RANGE_MAX_DEPTH) {
    t_range_stack[t_range_depth] = intern(message);
  }
  t_range_depth++; /* count past the maximum so pops stay balanced */
  ranges_recompute();
}

static void range_pop(void)
{
  if (t_range_depth > 0) {
    t_range_depth--;
  }
  ranges_recompute();
}

/* ─── Calls by correlation id ───────────────────────────────────────────────
 * Every tracked call leaves its sample weight and its ranges here, for the
 * buffer callback to filter, weigh and label its GPU records. Direct-mapped on
 * the low 32 bits of the sequential correlation ids, so it holds the last
 * CALL_SLOTS calls. */
#define CALL_SLOTS (1 << 20)

/* low 32 bits of the correlation id << 32 | sample weight, 0 if not sampled */
static uint64_t g_call_key[CALL_SLOTS];
static const char *g_call_ranges[CALL_SLOTS];

/* Timestamp at which tracing started, UINT64_MAX while it is off: the GPU work
 * of an untracked call that started earlier may come from before the profiler
 * attached. */
static uint64_t g_tracing_since = UINT64_MAX;
/* Whether GPU work is recorded: LAUNCH events are sent only then. */
static int g_tracing;

static void remember_call(uint64_t corr, uint32_t weight, const char *ranges)
{
  const uint32_t id = (uint32_t)corr;
  if (id == 0) {
    return;
  }
  const uint32_t h = id & (CALL_SLOTS - 1);
  __atomic_store_n(&g_call_ranges[h], ranges, __ATOMIC_RELEASE);
  __atomic_store_n(&g_call_key[h], (uint64_t)id << 32 | weight, __ATOMIC_RELEASE);
}

/* Returns the sample weight of the call corr, 0 if its GPU records are not
 * reported, with its ranges. The records of an untracked call, which sent no
 * LAUNCH event, count once if they started after tracing did. */
static uint32_t call_weight(uint64_t corr, uint64_t start, const char **ranges)
{
  *ranges           = NULL;
  const uint32_t id = (uint32_t)corr;
  if (id != 0) {
    const uint32_t h    = id & (CALL_SLOTS - 1);
    const uint64_t key  = __atomic_load_n(&g_call_key[h], __ATOMIC_ACQUIRE);
    const uint32_t call = (uint32_t)(key >> 32);
    if (key != 0 && call == id) {
      *ranges = __atomic_load_n(&g_call_ranges[h], __ATOMIC_ACQUIRE);
      return (uint32_t)key;
    }
    if (key != 0 && (int32_t)(call - id) > 0) {
      /* A later call took the slot: whether this one was sampled is lost. */
      static int reported;
      if (!__atomic_exchange_n(&reported, 1, __ATOMIC_RELAXED)) {
        log_error(-1, "call table too small: GPU records were dropped");
      }
      return 0;
    }
  }
  return start >= __atomic_load_n(&g_tracing_since, __ATOMIC_ACQUIRE) ? 1 : 0;
}

/* ─── Call sampling ──────────────────────────────────────────────────────────
 * Same classes and defaults as the CUDA producer. */
enum call_class {
  CALL_UNTRACKED,
  CALL_KERNEL,
  CALL_GRAPH,
  CALL_MEMORY,
  CALL_CLASSES,
};

#define DEFAULT_SAMPLE_RATE       200
#define DEFAULT_GRAPH_SAMPLE_RATE 20

/* Sampled calls per process per second, by class; 0 samples all of them. Each
 * class is sampled for the whole process, so that a process with many threads
 * is not reported more often than one with a single thread. */
static uint32_t g_sample_rate[CALL_CLASSES];
static struct otel_sampler g_samplers[CALL_CLASSES];
static pthread_mutex_t g_sampler_mu[CALL_CLASSES] = {
  PTHREAD_MUTEX_INITIALIZER,
  PTHREAD_MUTEX_INITIALIZER,
  PTHREAD_MUTEX_INITIALIZER,
  PTHREAD_MUTEX_INITIALIZER,
};
static __thread uint64_t t_rng;

static uint64_t now_ns(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static uint32_t sample_call(enum call_class cls)
{
  if (g_sample_rate[cls] == 0) {
    return 1;
  }
  const uint64_t now = now_ns();
  if (t_rng == 0) {
    t_rng = (now ^ (uint64_t)(uintptr_t)&t_rng) | 1;
  }
  pthread_mutex_lock(&g_sampler_mu[cls]);
  const uint32_t weight = otel_sampler_weight(&g_samplers[cls], g_sample_rate[cls], now, &t_rng);
  pthread_mutex_unlock(&g_sampler_mu[cls]);
  return weight;
}

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

/* Tracked HIP calls send LAUNCH events: those that launch kernels or graphs,
 * or copy or fill memory. ROCm versions number HIP calls differently and add
 * new ones, so they are found by name at start. The first matching prefix
 * decides. */
static const struct {
  const char *prefix;
  enum call_class cls;
} g_call_prefixes[] = {
  /* Host functions run on the CPU. */
  {"hipLaunchHostFunc", CALL_UNTRACKED},
  {"hipGraphLaunch", CALL_GRAPH},
  {"hipLaunch", CALL_KERNEL},
  {"hipExtLaunch", CALL_KERNEL},
  {"hipModuleLaunch", CALL_KERNEL},
  {"hipExtModuleLaunch", CALL_KERNEL},
  {"hipMemcpy", CALL_MEMORY},
  {"hipMemset", CALL_MEMORY},
};

/* The class of each HIP operation; filled at start. */
#define MAX_HIP_OPERATIONS 4096
static uint8_t g_operation_class[MAX_HIP_OPERATIONS];

/* The tracked operations this ROCm version has, found at start. */
struct tracked_operations {
  rocprofiler_tracing_operation_t ids[MAX_HIP_OPERATIONS];
  size_t count;
};

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

static int track_call(
  rocprofiler_callback_tracing_kind_t kind, rocprofiler_tracing_operation_t operation, void *data)
{
  struct tracked_operations *tracked = (struct tracked_operations *)data;
  const char *name                   = NULL;
  uint64_t len                       = 0;
  if (
    operation < 0 || operation >= MAX_HIP_OPERATIONS ||
    rocprofiler_query_callback_tracing_kind_operation_name(kind, operation, &name, &len) !=
      ROCPROFILER_STATUS_SUCCESS ||
    !name) {
    return 0;
  }
  const enum call_class cls = classify_call(name);
  if (cls != CALL_UNTRACKED) {
    g_operation_class[operation]   = (uint8_t)cls;
    tracked->ids[tracked->count++] = operation;
  }
  return 0;
}

static enum call_class call_class(rocprofiler_tracing_operation_t operation)
{
  if (operation < 0 || operation >= MAX_HIP_OPERATIONS) {
    return CALL_UNTRACKED;
  }
  return (enum call_class)g_operation_class[operation];
}

/* ─── Events ──────────────────────────────────────────────────────────────── */

/* emit_event is the only place firing the probe, so the profiler attaches one
 * uprobe to the library. */
OTEL_GPU_EMITTER static void emit_event(const struct otel_gpu_event *ev)
{
  OTEL_GPU_EVENT(ev);
}

static void
begin_event(struct otel_gpu_event *ev, enum otel_gpu_kind kind, uint64_t corr, uint32_t weight)
{
  ev->correlation_id = corr;
  ev->sample_weight  = weight;
  ev->kind           = (uint16_t)kind;
  ev->payload_size   = 0;
}

/* ─── GPU indexes ────────────────────────────────────────────────────────────
 * Events of work name their GPU by an index, its agent's position plus one; a
 * DEVICE event tells the profiler which UUID an index stands for. It is sent
 * before the first work on a GPU in each epoch: a new epoch starts when tracing
 * does, as the profiler may be a new one, and every GPU_RESEND_PERIOD_MS, in
 * case a DEVICE event was lost. */
#define GPU_RESEND_PERIOD_MS 10000

static uint32_t g_gpu_epoch;
static uint32_t g_gpu_sent_epoch[MAX_AGENTS];

/* Returns the index of a GPU agent, 0 for other agents, telling the profiler
 * its UUID first if needed. */
static uint32_t gpu_index(rocprofiler_agent_id_t agent)
{
  for (size_t i = 0; i < g_num_agents; i++) {
    if (g_agent_handle[i] != agent.handle) {
      continue;
    }
    if (!g_agent_uuid[i][0]) {
      return 0;
    }
    const uint32_t epoch = __atomic_load_n(&g_gpu_epoch, __ATOMIC_ACQUIRE);
    if (__atomic_load_n(&g_gpu_sent_epoch[i], __ATOMIC_ACQUIRE) != epoch) {
      struct otel_gpu_event ev;
      begin_event(&ev, OTEL_GPU_DEVICE, 0, 0);
      otel_gpu_put_string(&ev, OTEL_GPU_FIELD_GPU, g_agent_uuid[i]);
      otel_gpu_put_uint(&ev, OTEL_GPU_FIELD_GPU_INDEX, i + 1);
      emit_event(&ev);
      // Marked once sent: no other thread may report work on the GPU before it.
      __atomic_store_n(&g_gpu_sent_epoch[i], epoch, __ATOMIC_RELEASE);
    }
    return (uint32_t)i + 1;
  }
  return 0;
}

/* ─── Tracing while the profiler is attached ────────────────────────────────
 * GPU work is recorded only while the probe's semaphore is raised, so the
 * producer costs little without a profiler: a check of the semaphore at each
 * tracked call. */
static pthread_mutex_t g_tracing_mu = PTHREAD_MUTEX_INITIALIZER;
/* Set once tracing could not start: the producer then reports nothing. */
static int g_tracing_failed;

/* Follows the probe's semaphore: on when the profiler attached, off when it
 * left. */
static void update_tracing(void)
{
  const int want = OTEL_GPU_ENABLED() != 0;
  if (!g_enabled || want == __atomic_load_n(&g_tracing, __ATOMIC_ACQUIRE)) {
    return;
  }
  pthread_mutex_lock(&g_tracing_mu);
  if (want != g_tracing && !g_tracing_failed) {
    if (want) {
      rocprofiler_timestamp_t now = 0;
      rocprofiler_get_timestamp(&now);
      __atomic_store_n(&g_tracing_since, now, __ATOMIC_RELEASE);
      __atomic_add_fetch(&g_gpu_epoch, 1, __ATOMIC_ACQ_REL);
      const rocprofiler_status_t status = rocprofiler_start_context(g_trace_context);
      if (status == ROCPROFILER_STATUS_SUCCESS) {
        __atomic_store_n(&g_tracing, 1, __ATOMIC_RELEASE);
      } else {
        log_error((int)status, "starting tracing failed; GPU profiling disabled");
        __atomic_store_n(&g_tracing_since, UINT64_MAX, __ATOMIC_RELEASE);
        g_tracing_failed = 1;
      }
    } else {
      __atomic_store_n(&g_tracing, 0, __ATOMIC_RELEASE);
      __atomic_store_n(&g_tracing_since, UINT64_MAX, __ATOMIC_RELEASE);
      const rocprofiler_status_t status = rocprofiler_stop_context(g_trace_context);
      if (status != ROCPROFILER_STATUS_SUCCESS) {
        log_error((int)status, "stopping tracing failed");
      }
    }
  }
  pthread_mutex_unlock(&g_tracing_mu);
}

/* put_work adds the fields of an event of GPU work: the GPU, how long it took,
 * how many bytes it moved and the runtime, then the ranges at its call,
 * innermost first, as long as they fit whole, and last its name, cut short if
 * needed. */
static void put_work(
  struct otel_gpu_event *ev,
  uint32_t gpu,
  uint64_t start,
  uint64_t end,
  uint64_t bytes,
  const char *name,
  const char *ranges)
{
  otel_gpu_put_uint(ev, OTEL_GPU_FIELD_GPU_INDEX, gpu);
  otel_gpu_put_uint(ev, OTEL_GPU_FIELD_DURATION_NS, end > start ? end - start : 0);
  otel_gpu_put_uint(ev, OTEL_GPU_FIELD_BYTES, bytes);
  otel_gpu_put_uint(ev, OTEL_GPU_FIELD_RUNTIME, OTEL_GPU_RUNTIME_ROCM);
  for (const char *range = ranges; range && *range;) {
    const char *sep  = strchr(range, RANGE_SEP);
    const size_t len = sep ? (size_t)(sep - range) : strlen(range);
    if (!otel_gpu_put_whole_bytes(ev, OTEL_GPU_FIELD_RANGES, range, len)) {
      break;
    }
    range = sep ? sep + 1 : NULL;
  }
  otel_gpu_put_string(ev, OTEL_GPU_FIELD_NAME, name);
}

/* ─── Callbacks on the application's threads ─────────────────────────────── */

static void
hip_api_callback(rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t *, void *)
{
  if (record.phase != ROCPROFILER_CALLBACK_PHASE_ENTER) {
    return;
  }
  const enum call_class cls = call_class(record.operation);
  if (cls == CALL_UNTRACKED) {
    return;
  }
  const uint64_t corr = record.correlation_id.internal;

  // Sampled calls send a LAUNCH event, for the profiler to record the host
  // stack under the correlation id. None is sent while GPU work is not
  // recorded: the call's GPU records would never come.
  const int reporting   = OTEL_GPU_ENABLED() && __atomic_load_n(&g_tracing, __ATOMIC_ACQUIRE);
  const uint32_t weight = reporting ? sample_call(cls) : 0;
  if (weight != 0) {
    struct otel_gpu_event ev;
    begin_event(&ev, OTEL_GPU_LAUNCH, corr, weight);
    emit_event(&ev);
  }
  remember_call(corr, weight, weight != 0 ? t_ranges : NULL);
}

static void
marker_callback(rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t *, void *)
{
  if (record.phase != ROCPROFILER_CALLBACK_PHASE_ENTER) {
    return;
  }
  const rocprofiler_callback_tracing_marker_api_data_t *data =
    (const rocprofiler_callback_tracing_marker_api_data_t *)record.payload;
  switch (record.operation) {
  case ROCPROFILER_MARKER_CORE_API_ID_roctxRangePushA:
    range_push(data->args.roctxRangePushA.message);
    break;
  case ROCPROFILER_MARKER_CORE_API_ID_roctxRangePop: range_pop(); break;
  default: break;
  }
}

static void code_object_callback(
  rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t *, void *)
{
  if (
    record.operation != ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER ||
    record.phase != ROCPROFILER_CALLBACK_PHASE_LOAD) {
    return;
  }
  const rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t *data =
    (const rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t *)record.payload;
  register_kernel(data->kernel_id, data->kernel_name);
}

/* ─── GPU records ───────────────────────────────────────────────────────────── */

/* Returns the kind of event of a copy, 0 for copies not reported. */
static uint16_t copy_kind(rocprofiler_memory_copy_operation_t operation)
{
  switch (operation) {
  case ROCPROFILER_MEMORY_COPY_HOST_TO_DEVICE: return OTEL_GPU_COPY_HTOD;
  case ROCPROFILER_MEMORY_COPY_DEVICE_TO_HOST: return OTEL_GPU_COPY_DTOH;
  case ROCPROFILER_MEMORY_COPY_DEVICE_TO_DEVICE: return OTEL_GPU_COPY_DTOD;
  case ROCPROFILER_MEMORY_COPY_HOST_TO_HOST: return OTEL_GPU_COPY_HTOH;
  default: return 0;
  }
}

static void kernel_dispatch(const rocprofiler_buffer_tracing_kernel_dispatch_record_t *k)
{
  const uint64_t corr = k->correlation_id.internal;
  const char *ranges;
  const uint32_t weight = call_weight(corr, k->start_timestamp, &ranges);
  if (weight == 0) {
    return;
  }
  struct otel_gpu_event ev;
  begin_event(&ev, OTEL_GPU_KERNEL, corr, weight);
  put_work(
    &ev,
    gpu_index(k->dispatch_info.agent_id),
    k->start_timestamp,
    k->end_timestamp,
    0,
    kernel_name(k->dispatch_info.kernel_id),
    ranges);
  emit_event(&ev);
}

static void memory_copy(const rocprofiler_buffer_tracing_memory_copy_record_t *m)
{
  const uint64_t corr = m->correlation_id.internal;
  const char *ranges;
  const uint32_t weight = call_weight(corr, m->start_timestamp, &ranges);
  if (weight == 0) {
    return;
  }
  const uint16_t kind = copy_kind(m->operation);
  if (kind == 0) {
    return;
  }
  uint32_t gpu = gpu_index(m->dst_agent_id);
  if (gpu == 0) {
    gpu = gpu_index(m->src_agent_id);
  }
  struct otel_gpu_event ev;
  begin_event(&ev, (enum otel_gpu_kind)kind, corr, weight);
  put_work(&ev, gpu, m->start_timestamp, m->end_timestamp, m->bytes, NULL, ranges);
  emit_event(&ev);
}

static void buffer_callback(
  rocprofiler_context_id_t,
  rocprofiler_buffer_id_t,
  rocprofiler_record_header_t **headers,
  size_t num_headers,
  void *,
  uint64_t drop_count)
{
  static int reported;
  if (drop_count > 0 && !__atomic_exchange_n(&reported, 1, __ATOMIC_RELAXED)) {
    log_error(-1, "rocprofiler buffer full: GPU records were dropped");
  }
  for (size_t i = 0; i < num_headers; i++) {
    const rocprofiler_record_header_t *h = headers[i];
    if (h->category != ROCPROFILER_BUFFER_CATEGORY_TRACING) {
      continue;
    }
    switch (h->kind) {
    case ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH:
      kernel_dispatch((const rocprofiler_buffer_tracing_kernel_dispatch_record_t *)h->payload);
      break;
    case ROCPROFILER_BUFFER_TRACING_MEMORY_COPY:
      memory_copy((const rocprofiler_buffer_tracing_memory_copy_record_t *)h->payload);
      break;
    default: break;
    }
  }
}

/* Follows the profiler from here, not from the application's calls, which
 * would wait while tracing starts or stops, and hands over the records of
 * finished work: otherwise they sit until the buffer reaches its watermark. */
#define FLUSH_PERIOD_MS 500

static void *tracing_thread(void *)
{
  for (uint64_t round = 1;; round++) {
    update_tracing();
    if (__atomic_load_n(&g_tracing, __ATOMIC_ACQUIRE)) {
      rocprofiler_flush_buffer(g_buffer);
      if (round % (GPU_RESEND_PERIOD_MS / FLUSH_PERIOD_MS) == 0) {
        __atomic_add_fetch(&g_gpu_epoch, 1, __ATOMIC_ACQ_REL);
      }
    }
    usleep(FLUSH_PERIOD_MS * 1000);
  }
  return NULL;
}

/* ─── Setup ───────────────────────────────────────────────────────────────── */

#define CHECK(call, msg)                                                                           \
  do {                                                                                             \
    const rocprofiler_status_t status_ = (call);                                                   \
    if (status_ != ROCPROFILER_STATUS_SUCCESS) {                                                   \
      log_error((int)status_, msg);                                                                \
      return -1;                                                                                   \
    }                                                                                              \
  } while (0)

static int tool_init(rocprofiler_client_finalize_t, void *)
{
  g_sample_rate[CALL_KERNEL] = env_rate("OTEL_GPU_SAMPLE_RATE", DEFAULT_SAMPLE_RATE);
  g_sample_rate[CALL_MEMORY] = g_sample_rate[CALL_KERNEL];
  /* Sampling switched off covers graph launches too, unless they are set apart. */
  g_sample_rate[CALL_GRAPH]  = env_rate(
    "OTEL_GPU_GRAPH_SAMPLE_RATE", g_sample_rate[CALL_KERNEL] == 0 ? 0 : DEFAULT_GRAPH_SAMPLE_RATE);

  CHECK(
    rocprofiler_query_available_agents(
      ROCPROFILER_AGENT_INFO_VERSION_0, collect_agents, sizeof(rocprofiler_agent_v0_t), NULL),
    "rocprofiler_query_available_agents failed");
  CHECK(rocprofiler_create_context(&g_context), "rocprofiler_create_context failed");

  static struct tracked_operations tracked;
  CHECK(
    rocprofiler_iterate_callback_tracing_kind_operations(
      ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API, track_call, &tracked),
    "listing HIP calls failed");
  CHECK(
    rocprofiler_configure_callback_tracing_service(
      g_context,
      ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API,
      tracked.ids,
      tracked.count,
      hip_api_callback,
      NULL),
    "tracing HIP calls failed");
  CHECK(
    rocprofiler_configure_callback_tracing_service(
      g_context, ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT, NULL, 0, code_object_callback, NULL),
    "tracing code objects failed");
  rocprofiler_tracing_operation_t marker_operations[] = {
    ROCPROFILER_MARKER_CORE_API_ID_roctxRangePushA,
    ROCPROFILER_MARKER_CORE_API_ID_roctxRangePop,
  };
  CHECK(
    rocprofiler_configure_callback_tracing_service(
      g_context,
      ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API,
      marker_operations,
      sizeof(marker_operations) / sizeof(marker_operations[0]),
      marker_callback,
      NULL),
    "tracing ROCTx ranges failed");

  /* A full buffer drops records rather than blocking the application. */
  CHECK(rocprofiler_create_context(&g_trace_context), "rocprofiler_create_context failed");
  CHECK(
    rocprofiler_create_buffer(
      g_trace_context,
      BUFFER_SIZE,
      BUFFER_WATERMARK,
      ROCPROFILER_BUFFER_POLICY_DISCARD,
      buffer_callback,
      NULL,
      &g_buffer),
    "rocprofiler_create_buffer failed");
  CHECK(
    rocprofiler_configure_buffer_tracing_service(
      g_trace_context, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH, NULL, 0, g_buffer),
    "tracing kernel dispatches failed");
  CHECK(
    rocprofiler_configure_buffer_tracing_service(
      g_trace_context, ROCPROFILER_BUFFER_TRACING_MEMORY_COPY, NULL, 0, g_buffer),
    "tracing memory copies failed");

  rocprofiler_callback_thread_t thread;
  CHECK(rocprofiler_create_callback_thread(&thread), "rocprofiler_create_callback_thread failed");
  CHECK(
    rocprofiler_assign_callback_thread(g_buffer, thread),
    "rocprofiler_assign_callback_thread failed");
  CHECK(rocprofiler_start_context(g_context), "rocprofiler_start_context failed");

  g_enabled = 1;
  update_tracing();
  start_thread(tracing_thread);
  return 0;
}

static void tool_fini(void *)
{
  if (g_enabled) {
    rocprofiler_flush_buffer(g_buffer);
  }
}

static void fork_prepare(void)
{
  pthread_mutex_lock(&g_tracing_mu);
  pthread_mutex_lock(&g_pool_mu);
  for (size_t i = 0; i < CALL_CLASSES; i++) {
    pthread_mutex_lock(&g_sampler_mu[i]);
  }
}

static void fork_parent(void)
{
  for (size_t i = CALL_CLASSES; i > 0; i--) {
    pthread_mutex_unlock(&g_sampler_mu[i - 1]);
  }
  pthread_mutex_unlock(&g_pool_mu);
  pthread_mutex_unlock(&g_tracing_mu);
}

/* A child of a HIP process cannot use the GPU, nor flush the buffer at exit. */
static void fork_child(void)
{
  fork_parent();
  g_enabled = 0;
}

/* rocprofiler-sdk calls this entry point when it loads the library named in
 * ROCP_TOOL_LIBRARIES, as the HIP runtime starts. */
extern "C" rocprofiler_tool_configure_result_t *
rocprofiler_configure(uint32_t, const char *, uint32_t, rocprofiler_client_id_t *client_id)
{
  client_id->name = "otel_gpu";
  /* Hold our mutexes across fork: a child must never start with one locked by
   * a dead thread. */
  pthread_atfork(fork_prepare, fork_parent, fork_child);

  static rocprofiler_tool_configure_result_t result = {
    sizeof(rocprofiler_tool_configure_result_t), tool_init, tool_fini, NULL};
  return &result;
}
