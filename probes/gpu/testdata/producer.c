// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

// Fake otel_gpu producer for the integration tests: fires the USDT probes with
// deterministic records, so no GPU is needed. It waits up to argv[1] seconds
// for the profiler to attach, fires for argv[2] seconds, then
// prints what it fired, with values multiplied by their sample weights:
//   launches launch_weights kernels kernel_ns mem_ns mem_bytes alloc_bytes
//   uvm_bytes stalls collective_ns collective_bytes

#define OTEL_GPU_SEMAPHORES
#include "../otel_gpu_usdt.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

// Call sites rotate by correlation id: every tenth launches a CUDA graph of
// graphKernels kernels, a few others copy, set or allocate memory. Some stand
// for several call sites, as if sampled.
enum {
  graphKernels = 3,
  copyBytes    = 4096,
  setBytes     = 1024,
  allocBytes   = 1 << 20,
  graphWeight  = 2,
  copyWeight   = 3,
  kernelWeight = 4,
};

static const char vector_add[]    = "_Z10vector_addPfS_S_i";
static const char graph_step[]    = "_Z10graph_stepi";
static const char error[]         = "test producer error";
// Range stacks, from the innermost range outwards.
static const char matmul_ranges[] = "aten::mm\x1f"
                                    "aten::matmul\x1f"
                                    "train_step";
static const char step_range[]    = "train_step";

static const char stall_reason[] = "memory_dependency";
static const char all_reduce[]   = "AllReduce";

static unsigned long long launches, launch_weights, kernels, kernel_ns, mem_ns, mem_bytes,
  alloc_bytes, uvm_bytes, stalls, collective_ns, collective_bytes;

__attribute__((noinline)) static void launch(uint32_t correlation_id, uint32_t weight)
{
  const struct otel_gpu_launch_rec rec = {
    .correlation_id = correlation_id,
    .sample_weight  = weight,
  };
  OTEL_GPU_PROBE(on_launch, &rec);
  __asm__ volatile("" ::: "memory");
  launches++;
  launch_weights += weight;
}

static uint64_t duration_of(uint32_t correlation_id)
{
  return 1000 + correlation_id % 100;
}

static void
kernel_executed(uint32_t correlation_id, uint32_t weight, uint32_t graph_id, const char *name)
{
  const struct otel_gpu_kernel_rec rec = {
    .start          = 1000000,
    .end            = 1000000 + duration_of(correlation_id),
    .correlation_id = correlation_id,
    .graph_id       = graph_id,
    .name_ptr       = (uint64_t)name,
    .ranges_ptr     = correlation_id % 10 == 1 ? (uint64_t)matmul_ranges : 0,
    .sample_weight  = weight,
  };
  OTEL_GPU_PROBE(kernel_executed, &rec);
  kernels++;
  kernel_ns += weight * duration_of(correlation_id);
}

static void memcpy_executed(uint32_t correlation_id)
{
  const struct otel_gpu_mem_rec rec = {
    .start          = 1000000,
    .end            = 1000000 + duration_of(correlation_id),
    .correlation_id = correlation_id,
    .copy_kind      = OTEL_GPU_COPY_HTOD,
    .bytes          = copyBytes,
    .ranges_ptr     = (uint64_t)step_range,
    .sample_weight  = copyWeight,
  };
  OTEL_GPU_PROBE(gpu_mem, &rec);
  mem_ns += copyWeight * duration_of(correlation_id);
  mem_bytes += copyWeight * copyBytes;
}

static void memory_event(uint32_t correlation_id, uint32_t kind)
{
  struct otel_gpu_event_rec rec = {
    .start          = 1000000,
    .end            = 1000000 + duration_of(correlation_id),
    .kind           = kind,
    .correlation_id = correlation_id,
  };
  if (kind == OTEL_GPU_EVENT_MEMSET) {
    rec.v1 = setBytes;
    mem_ns += duration_of(correlation_id);
    mem_bytes += setBytes;
  } else {
    rec.v1 = allocBytes;
    rec.v2 = OTEL_GPU_MEMORY_DEVICE;
    alloc_bytes += allocBytes;
  }
  OTEL_GPU_PROBE(gpu_event, &rec);
}

static void pc_samples(void)
{
  const struct otel_gpu_event_rec rec = {
    .kind     = OTEL_GPU_EVENT_STALL,
    .name_ptr = (uint64_t)vector_add,
    .v1       = (uint64_t)stall_reason,
    .v2       = 5,
  };
  OTEL_GPU_PROBE(gpu_event, &rec);
  stalls += rec.v2;
}

static void collective_operation(uint32_t correlation_id)
{
  const struct otel_gpu_event_rec rec = {
    .start    = 1000000,
    .end      = 1000000 + duration_of(correlation_id),
    .kind     = OTEL_GPU_EVENT_COLLECTIVE,
    .v1       = 1 << 16,
    .name_ptr = (uint64_t)all_reduce,
  };
  OTEL_GPU_PROBE(gpu_event, &rec);
  collective_ns += duration_of(correlation_id);
  collective_bytes += rec.v1;
}

static void uvm_migration(void)
{
  const struct otel_gpu_event_rec rec = {.kind = OTEL_GPU_EVENT_UVM_HTOD, .v1 = 4096};
  OTEL_GPU_PROBE(gpu_event, &rec);
  uvm_bytes += rec.v1;
}

int main(int argc, char **argv)
{
  const time_t attach_deadline = time(NULL) + (argc > 1 ? atoi(argv[1]) : 10);
  OTEL_GPU_PROBE(ready, NULL);
  while (!OTEL_GPU_ENABLED(ready)) {
    if (time(NULL) >= attach_deadline) {
      fprintf(stderr, "the profiler did not attach\n");
      return 2;
    }
    usleep(10000);
  }

  const struct otel_gpu_error_rec failure = {.code = 15, .msg_ptr = (uint64_t)error};
  OTEL_GPU_PROBE(error, &failure);

  const time_t end        = time(NULL) + (argc > 2 ? atoi(argv[2]) : 3);
  uint32_t correlation_id = 0;
  while (time(NULL) < end) {
    correlation_id++;
    switch (correlation_id % 10) {
    case 0:
      launch(correlation_id, graphWeight);
      for (int i = 0; i < graphKernels; i++) {
        kernel_executed(correlation_id, graphWeight, 7, graph_step);
      }
      break;
    case 3:
      launch(correlation_id, copyWeight);
      memcpy_executed(correlation_id);
      break;
    case 6:
      launch(correlation_id, 1);
      memory_event(correlation_id, OTEL_GPU_EVENT_MEMSET);
      break;
    case 9:
      launch(correlation_id, 1);
      memory_event(correlation_id, OTEL_GPU_EVENT_ALLOC);
      uvm_migration();
      pc_samples();
      collective_operation(correlation_id);
      break;
    case 5:
      launch(correlation_id, kernelWeight);
      kernel_executed(correlation_id, kernelWeight, 0, vector_add);
      break;
    default: launch(correlation_id, 1); kernel_executed(correlation_id, 1, 0, vector_add);
    }
    usleep(1000);
  }
  printf(
    "%llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu\n",
    launches,
    launch_weights,
    kernels,
    kernel_ns,
    mem_ns,
    mem_bytes,
    alloc_bytes,
    uvm_bytes,
    stalls,
    collective_ns,
    collective_bytes);
  return 0;
}
