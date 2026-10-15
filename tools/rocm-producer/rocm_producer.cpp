// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0
//
// libotelrocm.so — rocprofiler-sdk tool library (ROCP_TOOL_LIBRARIES), a
// reference producer of the otel_gpu USDT probes for AMD GPUs, the ROCm
// counterpart of tools/cupti-producer. The contract is
// probes/gpu/otel_gpu_usdt.h.
//
// C++ only because the rocprofiler-sdk headers are: the code avoids the C++
// standard library, so the library does not depend on libstdc++ (it loads
// into arbitrary processes, where a second C++ runtime can clash).

#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#define OTEL_GPU_SEMAPHORES
#include "otel_gpu_usdt.h"
#include "sampler.h"

#define BUFFER_SIZE      (8u << 20)
#define BUFFER_WATERMARK (BUFFER_SIZE / 2)

static rocprofiler_context_id_t g_context;
static rocprofiler_buffer_id_t g_buffer;
static int g_enabled;

/* ─── Errors ─────────────────────────────────────────────────────────────────
 * An error fired at startup races the profiler attaching, so the last one is
 * re-emitted periodically and at exit. msg is static. */
static pthread_mutex_t g_err_mu = PTHREAD_MUTEX_INITIALIZER;
static struct otel_gpu_error_rec g_last_err;
static int g_err_started;

static void *error_reemit_thread(void *)
{
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

static void start_thread(void *(*fn)(void *))
{
  pthread_t t;
  pthread_attr_t a;
  pthread_attr_init(&a);
  pthread_attr_setdetachstate(&a, PTHREAD_CREATE_DETACHED);
  pthread_create(&t, &a, fn, NULL);
  pthread_attr_destroy(&a);
}

static void emit_error(int code, const char *msg)
{
  struct otel_gpu_error_rec rec = {};
  rec.code                      = code;
  rec.msg_ptr                   = (uint64_t)(uintptr_t)msg;
  OTEL_GPU_PROBE(error, &rec);
  pthread_mutex_lock(&g_err_mu);
  g_last_err = rec;
  if (!g_err_started) {
    g_err_started = 1;
    start_thread(error_reemit_thread);
  }
  pthread_mutex_unlock(&g_err_mu);
}

/* ─── Interned strings ──────────────────────────────────────────────────────
 * The profiler reads names from the process memory later, so every string
 * passed in a record is copied here once and never freed. Insert-only
 * open-addressed table: lock-free reads, inserts take the mutex. */
#define STRING_POOL_SLOTS 8192

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
  for (uint32_t i = 0; i < STRING_POOL_SLOTS; i++) {
    uint32_t slot = (h + i) & (STRING_POOL_SLOTS - 1);
    const char *p = __atomic_load_n(&g_pool[slot], __ATOMIC_ACQUIRE);
    if (p == NULL) {
      pthread_mutex_lock(&g_pool_mu);
      /* Re-probe under the lock: another thread may have inserted. */
      const char *found = NULL;
      for (; i < STRING_POOL_SLOTS; i++) {
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

/* ─── GPU indexes ───────────────────────────────────────────────────────────
 * Records name agents by opaque handle; the contract wants a device index. */
#define MAX_AGENTS 64

static uint64_t g_agent_handle[MAX_AGENTS];
static int32_t g_agent_gpu[MAX_AGENTS]; /* GPU index, -1 for other agents */
static size_t g_num_agents;

static rocprofiler_status_t
collect_agents(rocprofiler_agent_version_t, const void **agents, size_t num_agents, void *)
{
  for (size_t i = 0; i < num_agents && g_num_agents < MAX_AGENTS; i++) {
    const rocprofiler_agent_v0_t *a = (const rocprofiler_agent_v0_t *)agents[i];
    g_agent_handle[g_num_agents]    = a->id.handle;
    g_agent_gpu[g_num_agents] = a->type == ROCPROFILER_AGENT_TYPE_GPU ? a->logical_node_type_id : -1;
    g_num_agents++;
  }
  return ROCPROFILER_STATUS_SUCCESS;
}

static int32_t gpu_index(rocprofiler_agent_id_t agent)
{
  for (size_t i = 0; i < g_num_agents; i++) {
    if (g_agent_handle[i] == agent.handle) {
      return g_agent_gpu[i];
    }
  }
  return -1;
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
  t_ranges = NULL;
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

/* ─── Call sites by correlation id ──────────────────────────────────────────
 * Every tracked call site leaves its sample weight (0 if not sampled), its
 * ranges and whether it launches a graph here, for the buffer callback to
 * filter and label its GPU records. Direct-mapped on sequential correlation
 * ids, so it holds the last CALL_SITE_SLOTS calls. Correlation ids are 64-bit
 * in rocprofiler-sdk; the contract carries the low 32 bits. */
#define CALL_SITE_SLOTS (1 << 19)

static uint64_t g_site_key[CALL_SITE_SLOTS]; /* correlation id << 32 | weight */
static const char *g_site_ranges[CALL_SITE_SLOTS];
static uint8_t g_site_graph[CALL_SITE_SLOTS];

static void remember_call_site(uint32_t corr, uint32_t weight, const char *ranges, int graph)
{
  if (corr == 0) {
    return;
  }
  const uint32_t h = corr & (CALL_SITE_SLOTS - 1);
  __atomic_store_n(&g_site_ranges[h], ranges, __ATOMIC_RELEASE);
  __atomic_store_n(&g_site_graph[h], (uint8_t)graph, __ATOMIC_RELEASE);
  __atomic_store_n(&g_site_key[h], (uint64_t)corr << 32 | weight, __ATOMIC_RELEASE);
}

/* Returns the sample weight of the GPU records of call corr, 0 to drop them,
 * with its ranges and graph flag. Records of untracked calls count once. */
static uint32_t call_site_weight(uint32_t corr, const char **ranges, int *graph)
{
  *ranges = NULL;
  *graph  = 0;
  if (corr == 0) {
    return 1;
  }
  const uint32_t h    = corr & (CALL_SITE_SLOTS - 1);
  const uint64_t key  = __atomic_load_n(&g_site_key[h], __ATOMIC_ACQUIRE);
  const uint32_t site = (uint32_t)(key >> 32);
  if (key != 0 && site == corr) {
    *ranges = __atomic_load_n(&g_site_ranges[h], __ATOMIC_ACQUIRE);
    *graph  = __atomic_load_n(&g_site_graph[h], __ATOMIC_ACQUIRE);
    return (uint32_t)key;
  }
  if (key != 0 && (int32_t)(site - corr) > 0) {
    /* A later call took the slot: whether this one was sampled is lost. */
    static int reported;
    if (!__atomic_exchange_n(&reported, 1, __ATOMIC_RELAXED)) {
      emit_error(-1, "call site table too small: GPU records were dropped");
    }
    return 0;
  }
  return 1;
}

/* ─── Call site sampling ────────────────────────────────────────────────────
 * Same classes and defaults as the CUDA producer. */
enum call_site_class {
  SITE_KERNEL,
  SITE_GRAPH,
  SITE_MEMORY,
  SITE_ALLOC,
  SITE_CLASSES,
};

#define DEFAULT_SAMPLE_RATE       200
#define DEFAULT_GRAPH_SAMPLE_RATE 20

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

/* Tracked HIP calls: they fire on_launch. */
static const struct {
  rocprofiler_tracing_operation_t operation;
  enum call_site_class cls;
} g_call_sites[] = {
  {ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchKernel, SITE_KERNEL},
  {ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchKernelExC, SITE_KERNEL},
  {ROCPROFILER_HIP_RUNTIME_API_ID_hipExtLaunchKernel, SITE_KERNEL},
  {ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchCooperativeKernel, SITE_KERNEL},
  {ROCPROFILER_HIP_RUNTIME_API_ID_hipModuleLaunchKernel, SITE_KERNEL},
  {ROCPROFILER_HIP_RUNTIME_API_ID_hipExtModuleLaunchKernel, SITE_KERNEL},
  {ROCPROFILER_HIP_RUNTIME_API_ID_hipGraphLaunch, SITE_GRAPH},
  {ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy, SITE_MEMORY},
  {ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyAsync, SITE_MEMORY},
  {ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyWithStream, SITE_MEMORY},
  {ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyHtoD, SITE_MEMORY},
  {ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyHtoDAsync, SITE_MEMORY},
  {ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyDtoH, SITE_MEMORY},
  {ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyDtoHAsync, SITE_MEMORY},
  {ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyDtoD, SITE_MEMORY},
  {ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyDtoDAsync, SITE_MEMORY},
  {ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpy2DAsync, SITE_MEMORY},
  {ROCPROFILER_HIP_RUNTIME_API_ID_hipMemset, SITE_MEMORY},
  {ROCPROFILER_HIP_RUNTIME_API_ID_hipMemsetAsync, SITE_MEMORY},
  {ROCPROFILER_HIP_RUNTIME_API_ID_hipMemsetD8Async, SITE_MEMORY},
  {ROCPROFILER_HIP_RUNTIME_API_ID_hipMemsetD32Async, SITE_MEMORY},
  {ROCPROFILER_HIP_RUNTIME_API_ID_hipMalloc, SITE_ALLOC},
  {ROCPROFILER_HIP_RUNTIME_API_ID_hipMallocAsync, SITE_ALLOC},
  {ROCPROFILER_HIP_RUNTIME_API_ID_hipFree, SITE_ALLOC},
  {ROCPROFILER_HIP_RUNTIME_API_ID_hipFreeAsync, SITE_ALLOC},
};
#define NUM_CALL_SITES (sizeof(g_call_sites) / sizeof(g_call_sites[0]))

static int call_site_class(rocprofiler_tracing_operation_t operation)
{
  for (size_t i = 0; i < NUM_CALL_SITES; i++) {
    if (g_call_sites[i].operation == operation) {
      return g_call_sites[i].cls;
    }
  }
  return -1;
}

/* ─── Callbacks on the application's threads ─────────────────────────────── */

static void hip_api_callback(rocprofiler_callback_tracing_record_t record,
                             rocprofiler_user_data_t *,
                             void *)
{
  if (record.phase != ROCPROFILER_CALLBACK_PHASE_ENTER) {
    return;
  }
  const int cls = call_site_class(record.operation);
  if (cls < 0) {
    return;
  }
  const uint32_t corr = (uint32_t)record.correlation_id.internal;

  // Sampled call sites fire on_launch, the host stack keyed by correlation id.
  // Without a profiler attached none is: its GPU records would lack the stack.
  const uint32_t weight =
    OTEL_GPU_ENABLED(ready) ? sample_call_site((enum call_site_class)cls) : 0;
  if (weight != 0) {
    struct otel_gpu_launch_rec rec = {};
    rec.correlation_id             = corr;
    rec.sample_weight              = weight;
    OTEL_GPU_PROBE(on_launch, &rec);
  }
  remember_call_site(corr, weight, weight != 0 ? t_ranges : NULL, cls == SITE_GRAPH);
}

static void marker_callback(rocprofiler_callback_tracing_record_t record,
                            rocprofiler_user_data_t *,
                            void *)
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

static void code_object_callback(rocprofiler_callback_tracing_record_t record,
                                 rocprofiler_user_data_t *,
                                 void *)
{
  if (record.operation != ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER ||
      record.phase != ROCPROFILER_CALLBACK_PHASE_LOAD) {
    return;
  }
  const rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t *data =
    (const rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t *)record.payload;
  register_kernel(data->kernel_id, data->kernel_name);
}

/* ─── GPU records ───────────────────────────────────────────────────────────── */

static uint32_t copy_kind(rocprofiler_memory_copy_operation_t operation)
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
  const uint32_t corr = (uint32_t)k->correlation_id.internal;
  const char *ranges;
  int graph;
  const uint32_t weight = call_site_weight(corr, &ranges, &graph);
  if (weight == 0) {
    return;
  }
  const char *name               = kernel_name(k->dispatch_info.kernel_id);
  struct otel_gpu_kernel_rec rec = {};
  rec.start                      = k->start_timestamp;
  rec.end                        = k->end_timestamp;
  rec.correlation_id             = corr;
  rec.device_id                  = (uint32_t)gpu_index(k->dispatch_info.agent_id);
  rec.stream_id                  = (uint32_t)k->dispatch_info.queue_id.handle;
  // HIP does not report graph ids: any nonzero id keeps the launch for the
  // other kernels of the graph.
  rec.graph_id                   = graph ? 1 : 0;
  rec.name_ptr                   = (uint64_t)(uintptr_t)(name ? name : "");
  rec.ranges_ptr                 = (uint64_t)(uintptr_t)ranges;
  rec.sample_weight              = weight;
  OTEL_GPU_PROBE(kernel_executed, &rec);
}

static void memory_copy(const rocprofiler_buffer_tracing_memory_copy_record_t *m)
{
  const uint32_t corr = (uint32_t)m->correlation_id.internal;
  const char *ranges;
  int graph;
  const uint32_t weight = call_site_weight(corr, &ranges, &graph);
  if (weight == 0) {
    return;
  }
  const int32_t dst           = gpu_index(m->dst_agent_id);
  struct otel_gpu_mem_rec rec = {};
  rec.start                   = m->start_timestamp;
  rec.end                     = m->end_timestamp;
  rec.correlation_id          = corr;
  rec.device_id               = (uint32_t)(dst >= 0 ? dst : gpu_index(m->src_agent_id));
  rec.copy_kind               = copy_kind(m->operation);
  rec.bytes                   = m->bytes;
  rec.ranges_ptr              = (uint64_t)(uintptr_t)ranges;
  rec.sample_weight           = weight;
  OTEL_GPU_PROBE(gpu_mem, &rec);
}

static void memory_allocation(const rocprofiler_buffer_tracing_memory_allocation_record_t *a)
{
  uint32_t kind;
  switch (a->operation) {
  case ROCPROFILER_MEMORY_ALLOCATION_ALLOCATE:
  case ROCPROFILER_MEMORY_ALLOCATION_VMEM_ALLOCATE: kind = OTEL_GPU_EVENT_ALLOC; break;
  case ROCPROFILER_MEMORY_ALLOCATION_FREE:
  case ROCPROFILER_MEMORY_ALLOCATION_VMEM_FREE: kind = OTEL_GPU_EVENT_FREE; break;
  default: return;
  }
  const uint32_t corr = (uint32_t)a->correlation_id.internal;
  const char *ranges;
  int graph;
  const uint32_t weight = call_site_weight(corr, &ranges, &graph);
  if (weight == 0) {
    return;
  }
  struct otel_gpu_event_rec ev = {};
  ev.start                     = a->start_timestamp;
  ev.end                       = a->end_timestamp;
  ev.kind                      = kind;
  ev.correlation_id            = corr;
  ev.v1                        = a->allocation_size;
  ev.v2 = gpu_index(a->agent_id) >= 0 ? OTEL_GPU_MEMORY_DEVICE : OTEL_GPU_MEMORY_PINNED;
  ev.sample_weight = weight;
  OTEL_GPU_PROBE(gpu_event, &ev);
}

static void buffer_callback(rocprofiler_context_id_t,
                            rocprofiler_buffer_id_t,
                            rocprofiler_record_header_t **headers,
                            size_t num_headers,
                            void *,
                            uint64_t drop_count)
{
  if (drop_count > 0) {
    emit_error(-1, "rocprofiler buffer full: GPU records were dropped");
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
    case ROCPROFILER_BUFFER_TRACING_MEMORY_ALLOCATION:
      memory_allocation(
        (const rocprofiler_buffer_tracing_memory_allocation_record_t *)h->payload);
      break;
    default: break;
    }
  }
}

/* Otherwise records sit until the buffer reaches its watermark, delaying
 * export and aging out the call site table. */
static void *flush_thread(void *)
{
  for (;;) {
    sleep(1);
    rocprofiler_flush_buffer(g_buffer);
  }
  return NULL;
}

/* ─── Setup ───────────────────────────────────────────────────────────────── */

#define CHECK(call, msg)                                                                           \
  do {                                                                                             \
    const rocprofiler_status_t status_ = (call);                                                   \
    if (status_ != ROCPROFILER_STATUS_SUCCESS) {                                                   \
      emit_error((int)status_, msg);                                                               \
      return -1;                                                                                   \
    }                                                                                              \
  } while (0)

static int tool_init(rocprofiler_client_finalize_t, void *)
{
  g_sample_rate[SITE_KERNEL] = env_rate("OTEL_GPU_SAMPLE_RATE", DEFAULT_SAMPLE_RATE);
  g_sample_rate[SITE_MEMORY] = g_sample_rate[SITE_KERNEL];
  g_sample_rate[SITE_ALLOC]  = g_sample_rate[SITE_KERNEL];
  /* Sampling switched off covers graph launches too, unless they are set apart. */
  g_sample_rate[SITE_GRAPH]  = env_rate(
    "OTEL_GPU_GRAPH_SAMPLE_RATE", g_sample_rate[SITE_KERNEL] == 0 ? 0 : DEFAULT_GRAPH_SAMPLE_RATE);

  CHECK(rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0, collect_agents,
                                           sizeof(rocprofiler_agent_v0_t), NULL),
        "rocprofiler_query_available_agents failed");
  CHECK(rocprofiler_create_context(&g_context), "rocprofiler_create_context failed");

  rocprofiler_tracing_operation_t operations[NUM_CALL_SITES];
  for (size_t i = 0; i < NUM_CALL_SITES; i++) {
    operations[i] = g_call_sites[i].operation;
  }
  CHECK(rocprofiler_configure_callback_tracing_service(g_context,
                                                       ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API,
                                                       operations, NUM_CALL_SITES,
                                                       hip_api_callback, NULL),
        "tracing HIP calls failed");
  CHECK(rocprofiler_configure_callback_tracing_service(
          g_context, ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT, NULL, 0, code_object_callback, NULL),
        "tracing code objects failed");
  rocprofiler_tracing_operation_t marker_operations[] = {
    ROCPROFILER_MARKER_CORE_API_ID_roctxRangePushA,
    ROCPROFILER_MARKER_CORE_API_ID_roctxRangePop,
  };
  CHECK(rocprofiler_configure_callback_tracing_service(
          g_context, ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API, marker_operations,
          sizeof(marker_operations) / sizeof(marker_operations[0]), marker_callback, NULL),
        "tracing ROCTx ranges failed");

  CHECK(rocprofiler_create_buffer(g_context, BUFFER_SIZE, BUFFER_WATERMARK,
                                  ROCPROFILER_BUFFER_POLICY_LOSSLESS, buffer_callback, NULL,
                                  &g_buffer),
        "rocprofiler_create_buffer failed");
  CHECK(rocprofiler_configure_buffer_tracing_service(
          g_context, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH, NULL, 0, g_buffer),
        "tracing kernel dispatches failed");
  CHECK(rocprofiler_configure_buffer_tracing_service(
          g_context, ROCPROFILER_BUFFER_TRACING_MEMORY_COPY, NULL, 0, g_buffer),
        "tracing memory copies failed");
  CHECK(rocprofiler_configure_buffer_tracing_service(
          g_context, ROCPROFILER_BUFFER_TRACING_MEMORY_ALLOCATION, NULL, 0, g_buffer),
        "tracing memory allocations failed");

  rocprofiler_callback_thread_t thread;
  CHECK(rocprofiler_create_callback_thread(&thread), "rocprofiler_create_callback_thread failed");
  CHECK(rocprofiler_assign_callback_thread(g_buffer, thread),
        "rocprofiler_assign_callback_thread failed");
  CHECK(rocprofiler_start_context(g_context), "rocprofiler_start_context failed");

  start_thread(flush_thread);
  g_enabled = 1;
  return 0;
}

static void tool_fini(void *)
{
  if (g_enabled) {
    rocprofiler_flush_buffer(g_buffer);
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
  pthread_mutex_lock(&g_pool_mu);
}

static void fork_parent(void)
{
  pthread_mutex_unlock(&g_pool_mu);
  pthread_mutex_unlock(&g_err_mu);
}

/* rocprofiler-sdk calls this entry point when it loads the library named in
 * ROCP_TOOL_LIBRARIES, as the HIP runtime starts. */
extern "C" rocprofiler_tool_configure_result_t *
rocprofiler_configure(uint32_t, const char *, uint32_t, rocprofiler_client_id_t *client_id)
{
  client_id->name = "otel_gpu";
  /* Hold our mutexes across fork: a child must never start with one locked by
   * a dead thread. */
  pthread_atfork(fork_prepare, fork_parent, fork_parent);
  OTEL_GPU_PROBE(ready, NULL);

  static rocprofiler_tool_configure_result_t result = {
    sizeof(rocprofiler_tool_configure_result_t), tool_init, tool_fini, NULL};
  return &result;
}
