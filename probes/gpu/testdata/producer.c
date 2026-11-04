// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

// Fake otel_gpu_v1 producer for the integration tests: fires the USDT probe
// with deterministic events, so no GPU is needed. It waits up to argv[1]
// seconds for the profiler to attach, fires for argv[2] seconds, then prints
// what it fired, with values multiplied by the sample weight of their call,
// and the number of calls whose broken events the profiler must drop:
//   launches launch_weights kernels kernel_ns mem_ns mem_bytes broken
// On stderr it prints the highest value its semaphore took.
//
// It tells the profiler its GPU at start and again every second, as producers
// do every GPU_RESEND_PERIOD_MS. argv[3], if given, changes that:
//   lose-device  the first DEVICE event is lost
//   exec         after a second, execs itself as after-exec
//   after-exec   names another GPU by the same index, and reports its work
//                under kernel after_exec
//   busy         spins between calls instead of sleeping, so that a profiler
//                started later samples it on a CPU soon

#define OTEL_GPU_SEMAPHORE
#include "../otel_gpu_v1_usdt.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

// Calls rotate by call id: every tenth launches a CUDA graph of graphKernels
// kernels, a few others copy or fill memory, and one kernel runs without a
// call of its own. Some stand for several calls, as if sampled.
enum {
  graphKernels = 3,
  copyBytes    = 4096,
  fillBytes    = 1024,
  graphWeight  = 2,
  copyWeight   = 3,
  kernelWeight = 4,
  fillWeight   = 2,
};

static const char vector_add[]      = "_Z10vector_addPfS_S_i";
static const char graph_step[]      = "_Z10graph_stepi";
static const char device_launched[] = "_Z15device_launchedv";
static const char after_exec[]      = "_Z10after_execv";
static const char *gpu_uuid         = "GPU-00000000-0000-0000-0000-000000000001";
static const char exec_gpu_uuid[]   = "GPU-00000000-0000-0000-0000-000000000002";
// The index the events of work name the GPU by.
enum { gpuIndex = 1 };

// Call ids use all 64 bits, while the profiler tells calls apart by their low
// 32 bits.
static const uint64_t first_call = 1ull << 40;

static unsigned long long launches, launch_weights, kernels, kernel_ns, mem_ns, mem_bytes, broken;
static unsigned short semaphore_max;

// between_calls waits a millisecond between calls, spinning if busy.
static void between_calls(int busy)
{
  if (!busy) {
    usleep(1000);
    return;
  }
  struct timespec start, now;
  clock_gettime(CLOCK_MONOTONIC, &start);
  do {
    clock_gettime(CLOCK_MONOTONIC, &now);
  } while ((now.tv_sec - start.tv_sec) * 1000000000L + (now.tv_nsec - start.tv_nsec) < 1000000L);
}

static void watch_semaphore(void)
{
  if (otel_gpu_v1_event_semaphore > semaphore_max) {
    semaphore_max = otel_gpu_v1_event_semaphore;
  }
}

// emit is the only place firing the probe, so the profiler attaches one uprobe.
OTEL_GPU_EMITTER static void emit(const struct otel_gpu_event *ev)
{
  OTEL_GPU_EVENT(ev);
  __asm__ volatile("" ::: "memory");
}

static void begin(struct otel_gpu_event *ev, uint16_t kind, uint64_t call, uint32_t weight)
{
  ev->correlation_id = call;
  ev->sample_weight  = weight;
  ev->kind           = kind;
  ev->payload_size   = 0;
}

static uint64_t duration_of(uint64_t call)
{
  return 1000 + call % 100;
}

// device tells the profiler which GPU gpuIndex stands for.
static void device(void)
{
  struct otel_gpu_event ev;
  begin(&ev, OTEL_GPU_DEVICE, 0, 0);
  otel_gpu_put_string(&ev, OTEL_GPU_FIELD_GPU, gpu_uuid);
  otel_gpu_put_uint(&ev, OTEL_GPU_FIELD_GPU_INDEX, gpuIndex);
  emit(&ev);
}

// put_work adds the fields every event of work carries: the GPU, how long it
// took and the runtime.
static void put_work(struct otel_gpu_event *ev, uint64_t call)
{
  otel_gpu_put_uint(ev, OTEL_GPU_FIELD_GPU_INDEX, gpuIndex);
  otel_gpu_put_uint(ev, OTEL_GPU_FIELD_DURATION_NS, duration_of(call));
  otel_gpu_put_uint(ev, OTEL_GPU_FIELD_RUNTIME, OTEL_GPU_RUNTIME_CUDA);
}

static void put_range(struct otel_gpu_event *ev, const char *name)
{
  otel_gpu_put_whole_bytes(ev, OTEL_GPU_FIELD_RANGES, name, strlen(name));
}

static void launch(uint64_t call, uint32_t weight)
{
  struct otel_gpu_event ev;
  begin(&ev, OTEL_GPU_LAUNCH, call, weight);
  emit(&ev);
  launches++;
  launch_weights += weight;
}

static void kernel_executed(uint64_t call, uint32_t weight, const char *name)
{
  struct otel_gpu_event ev;
  begin(&ev, OTEL_GPU_KERNEL, call, weight);
  put_work(&ev, call);
  if (call % 10 == 1) {
    put_range(&ev, "aten::mm");
    put_range(&ev, "aten::matmul");
    put_range(&ev, "train_step");
  }
  otel_gpu_put_string(&ev, OTEL_GPU_FIELD_NAME, name);
  emit(&ev);
  kernels++;
  kernel_ns += weight * duration_of(call);
}

static void memory_operation(uint64_t call, uint16_t kind, uint32_t weight, uint64_t bytes)
{
  struct otel_gpu_event ev;
  begin(&ev, kind, call, weight);
  put_work(&ev, call);
  otel_gpu_put_uint(&ev, OTEL_GPU_FIELD_BYTES, bytes);
  put_range(&ev, "train_step");
  emit(&ev);
  mem_ns += weight * duration_of(call);
  mem_bytes += weight * bytes;
}

// broken_events sends events of a call that the profiler must drop: one whose
// payload does not decode, one whose payload is larger than allowed, and one
// that cannot be read.
static void broken_events(uint64_t call)
{
  broken++;
  struct otel_gpu_event ev;
  begin(&ev, OTEL_GPU_KERNEL, call, 1);
  // A string field that claims 16 bytes but has 1.
  ev.payload[0]   = OTEL_GPU_FIELD_NAME << 3 | 2;
  ev.payload[1]   = 16;
  ev.payload[2]   = 'k';
  ev.payload_size = 3;
  emit(&ev);

  begin(&ev, OTEL_GPU_KERNEL, call, 1);
  ev.payload_size = OTEL_GPU_PAYLOAD_MAX + 1;
  emit(&ev);

  // The first page is never mapped.
  emit((const struct otel_gpu_event *)(uintptr_t)8);
}

int main(int argc, char **argv)
{
  const time_t attach_deadline = time(NULL) + (argc > 1 ? atoi(argv[1]) : 10);
  const int seconds            = argc > 2 ? atoi(argv[2]) : 3;
  const char *mode             = argc > 3 ? argv[3] : "";
  while (!OTEL_GPU_ENABLED()) {
    if (time(NULL) >= attach_deadline) {
      fprintf(stderr, "the profiler did not attach\n");
      return 2;
    }
    usleep(10000);
  }

  const time_t start = time(NULL);
  const time_t end   = start + seconds;
  const int busy     = strcmp(mode, "busy") == 0;
  time_t next_device = strcmp(mode, "lose-device") == 0 ? start + 1 : start;
  if (strcmp(mode, "after-exec") == 0) {
    gpu_uuid = exec_gpu_uuid;
  }
  uint64_t call = first_call;
  while (time(NULL) < end) {
    watch_semaphore();
    if (time(NULL) >= next_device) {
      device();
      next_device = time(NULL) + 1;
    }
    if (strcmp(mode, "exec") == 0 && time(NULL) > start) {
      char remaining[16];
      snprintf(remaining, sizeof(remaining), "%d", (int)(end - time(NULL)));
      execl(argv[0], argv[0], "0", remaining, "after-exec", (char *)NULL);
      return 3;
    }
    call++;
    if (strcmp(mode, "after-exec") == 0) {
      launch(call, 1);
      kernel_executed(call, 1, after_exec);
      between_calls(busy);
      continue;
    }
    switch (call % 10) {
    case 0:
      launch(call, graphWeight);
      for (int i = 0; i < graphKernels; i++) {
        kernel_executed(call, graphWeight, graph_step);
      }
      break;
    case 3:
      launch(call, copyWeight);
      memory_operation(call, OTEL_GPU_COPY_HTOD, copyWeight, copyBytes);
      break;
    case 7:
      // A kernel no call started, as when kernels launch kernels.
      kernel_executed(0, 1, device_launched);
      break;
    case 6:
      launch(call, fillWeight);
      memory_operation(call, OTEL_GPU_FILL, fillWeight, fillBytes);
      break;
    case 9:
      launch(call, 1);
      broken_events(call);
      break;
    case 5:
      launch(call, kernelWeight);
      kernel_executed(call, kernelWeight, vector_add);
      break;
    default: launch(call, 1); kernel_executed(call, 1, vector_add);
    }
    between_calls(busy);
  }
  fprintf(stderr, "semaphore_max %u\n", semaphore_max);
  printf(
    "%llu %llu %llu %llu %llu %llu %llu\n",
    launches,
    launch_weights,
    kernels,
    kernel_ns,
    mem_ns,
    mem_bytes,
    broken);
  return 0;
}
