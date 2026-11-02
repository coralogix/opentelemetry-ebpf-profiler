// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

// Fake otel_gpu_v1 producer for the integration tests: fires the USDT probe
// with deterministic events, so no GPU is needed. It waits up to argv[1]
// seconds for the profiler to attach, fires for argv[2] seconds, then prints
// what it fired, with values multiplied by the sample weight of their call,
// and the number of calls whose broken events the profiler must drop:
//   launches launch_weights kernels kernel_ns mem_ns mem_bytes broken

#define OTEL_GPU_SEMAPHORE
#include "../otel_gpu_v1_usdt.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

// Calls rotate by call id: every tenth launches a CUDA graph of graphKernels
// kernels, a few others copy or fill memory. Some stand for several calls, as
// if sampled.
enum {
  graphKernels = 3,
  copyBytes    = 4096,
  fillBytes    = 1024,
  graphWeight  = 2,
  copyWeight   = 3,
  kernelWeight = 4,
  fillWeight   = 2,
};

static const char vector_add[] = "_Z10vector_addPfS_S_i";
static const char graph_step[] = "_Z10graph_stepi";
static const char gpu_uuid[]   = "GPU-00000000-0000-0000-0000-000000000001";

// Call ids use all 64 bits, while the profiler tells calls apart by their low
// 32 bits.
static const uint64_t first_call = 1ull << 40;

static unsigned long long launches, launch_weights, kernels, kernel_ns, mem_ns, mem_bytes, broken;

// emit is the only place firing the probe, so the profiler attaches one uprobe.
__attribute__((noinline)) static void emit(const struct otel_gpu_event *ev)
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

static void launch(uint64_t call, uint32_t weight)
{
  struct otel_gpu_event ev;
  begin(&ev, OTEL_GPU_LAUNCH, call, weight);
  emit(&ev);
  launches++;
  launch_weights += weight;
}

static uint64_t duration_of(uint64_t call)
{
  return 1000 + call % 100;
}

static void kernel_executed(uint64_t call, uint32_t weight, const char *name)
{
  struct otel_gpu_event ev;
  begin(&ev, OTEL_GPU_KERNEL, call, weight);
  otel_gpu_put_string(&ev, OTEL_GPU_FIELD_GPU, gpu_uuid);
  otel_gpu_put_string(&ev, OTEL_GPU_FIELD_NAME, name);
  if (call % 10 == 1) {
    otel_gpu_put_string(&ev, OTEL_GPU_FIELD_RANGES, "aten::mm");
    otel_gpu_put_string(&ev, OTEL_GPU_FIELD_RANGES, "aten::matmul");
    otel_gpu_put_string(&ev, OTEL_GPU_FIELD_RANGES, "train_step");
  }
  otel_gpu_put_uint(&ev, OTEL_GPU_FIELD_DURATION_NS, duration_of(call));
  emit(&ev);
  kernels++;
  kernel_ns += weight * duration_of(call);
}

static void memory_operation(uint64_t call, uint16_t kind, uint32_t weight, uint64_t bytes)
{
  struct otel_gpu_event ev;
  begin(&ev, kind, call, weight);
  otel_gpu_put_string(&ev, OTEL_GPU_FIELD_GPU, gpu_uuid);
  if (kind == OTEL_GPU_COPY) {
    otel_gpu_put_string(&ev, OTEL_GPU_FIELD_NAME, "HtoD");
  }
  otel_gpu_put_string(&ev, OTEL_GPU_FIELD_RANGES, "train_step");
  otel_gpu_put_uint(&ev, OTEL_GPU_FIELD_DURATION_NS, duration_of(call));
  otel_gpu_put_uint(&ev, OTEL_GPU_FIELD_BYTES, bytes);
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
  while (!OTEL_GPU_ENABLED()) {
    if (time(NULL) >= attach_deadline) {
      fprintf(stderr, "the profiler did not attach\n");
      return 2;
    }
    usleep(10000);
  }

  const time_t end = time(NULL) + (argc > 2 ? atoi(argv[2]) : 3);
  uint64_t call    = first_call;
  while (time(NULL) < end) {
    call++;
    switch (call % 10) {
    case 0:
      launch(call, graphWeight);
      for (int i = 0; i < graphKernels; i++) {
        kernel_executed(call, graphWeight, graph_step);
      }
      break;
    case 3:
      launch(call, copyWeight);
      memory_operation(call, OTEL_GPU_COPY, copyWeight, copyBytes);
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
    usleep(1000);
  }
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
