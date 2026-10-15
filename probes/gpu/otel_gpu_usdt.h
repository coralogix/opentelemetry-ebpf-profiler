// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

// USDT contract between GPU telemetry producers, such as a CUPTI injection
// library, and the gpu probe of the OpenTelemetry eBPF profiler.
//
// Provider "otel_gpu". Every probe passes one pointer to a fixed-layout record,
// pinned to a register ("8@%rax" on x86-64, "8@x0" on arm64). Without pinning,
// compilers pick a register per call site and multi-argument probes turn into
// memory operands that change between builds. The profiler checks the
// argument spec when attaching and refuses any other spec.
//
// Strings are passed as pointers into the producer's memory and must stay
// valid for the lifetime of the process. A range stack lists the names of the
// annotation ranges (such as NVTX) active at a call site, from the innermost
// outwards, separated by 0x1f. Breaking changes ship as new probe names with a
// _v2 suffix.
//
// Producers may sample call sites: on_launch then fires for sampled call sites
// only, and the GPU records of the other call sites are not fired at all. The
// launch record and the GPU records of a sampled call site carry its sample
// weight, the number of call sites it stands for; the profiler multiplies the
// values of the GPU records by it. A weight of 0 counts as 1.
//
// Producers should profile call sites only while a consumer is attached: GPU
// records of earlier call sites would arrive without a launch stack. Defining
// OTEL_GPU_SEMAPHORES before including this header, in one translation unit,
// gives each probe a semaphore that the kernel raises while a consumer is
// attached. The profiler attaches to ready last, once every other probe works,
// so producers profile while OTEL_GPU_ENABLED(ready) is true.
#pragma once

#include <assert.h>
#include <stdint.h>

#ifdef OTEL_GPU_SEMAPHORES
  #define _SDT_HAS_SEMAPHORES 1
  // Semaphores must be file-backed for the kernel to update them, hence .probes.
  #define OTEL_GPU_SEMAPHORE(name)                                                                 \
    __attribute__((                                                                                \
      section(".probes"),                                                                          \
      visibility("hidden"))) volatile unsigned short otel_gpu_##name##_semaphore
OTEL_GPU_SEMAPHORE(ready);
OTEL_GPU_SEMAPHORE(on_launch);
OTEL_GPU_SEMAPHORE(kernel_executed);
OTEL_GPU_SEMAPHORE(gpu_mem);
OTEL_GPU_SEMAPHORE(gpu_event);
OTEL_GPU_SEMAPHORE(error);
  #define OTEL_GPU_ENABLED(name) (otel_gpu_##name##_semaphore != 0)
#endif

#include <sys/sdt.h>

#if defined(__x86_64__)
  #define OTEL_GPU_RECORD_REGISTER "rax"
#elif defined(__aarch64__)
  #define OTEL_GPU_RECORD_REGISTER "x0"
#else
  #error "otel_gpu USDT probes support x86-64 and arm64 only"
#endif

// OTEL_GPU_PROBE fires probe `name` of provider otel_gpu with record `rec`.
#define OTEL_GPU_PROBE(name, rec)                                                                  \
  do {                                                                                             \
    register const void *otel_gpu_record_ asm(OTEL_GPU_RECORD_REGISTER) = (rec);                   \
    DTRACE_PROBE1(otel_gpu, name, otel_gpu_record_);                                               \
  } while (0)

// ready: fired once when the producer starts, with a NULL record. Its
// semaphore tells the producer that the profiler is attached.

// on_launch: fired on the host thread at every tracked call site (kernel
// launches, memory copies and sets, allocations), or every sampled one. The
// profiler captures the host stack here and joins it with later GPU records by
// correlation_id.
struct otel_gpu_launch_rec {
  uint32_t correlation_id;
  uint32_t sample_weight;
};

// kernel_executed: one kernel execution measured on the GPU.
struct otel_gpu_kernel_rec {
  uint64_t start;          // GPU timestamp, ns
  uint64_t end;            // GPU timestamp, ns
  uint32_t correlation_id; // matches the on_launch record of the launch
  uint32_t device_id;
  uint32_t stream_id;
  uint32_t graph_id;   // CUDA graph id, 0 if not launched from a graph
  uint64_t name_ptr;   // const char *: kernel name, possibly mangled
  uint64_t ranges_ptr; // const char *: range stack at launch, or 0
  uint32_t sample_weight;
  uint32_t _pad;
};

// gpu_mem: one memory copy measured on the GPU.
struct otel_gpu_mem_rec {
  uint64_t start;          // GPU timestamp, ns
  uint64_t end;            // GPU timestamp, ns
  uint32_t correlation_id; // matches the on_launch record of the call
  uint32_t device_id;
  uint32_t stream_id;
  uint32_t copy_kind; // enum otel_gpu_copy_kind
  uint64_t bytes;
  uint64_t ranges_ptr; // const char *: range stack at the call, or 0
  uint32_t sample_weight;
  uint32_t _pad;
};

// Copy directions, matching CUpti_ActivityMemcpyKind.
enum otel_gpu_copy_kind {
  OTEL_GPU_COPY_HTOD = 1,
  OTEL_GPU_COPY_DTOH = 2,
  OTEL_GPU_COPY_HTOA = 3,
  OTEL_GPU_COPY_ATOH = 4,
  OTEL_GPU_COPY_ATOA = 5,
  OTEL_GPU_COPY_ATOD = 6,
  OTEL_GPU_COPY_DTOA = 7,
  OTEL_GPU_COPY_DTOD = 8,
  OTEL_GPU_COPY_HTOH = 9,
  OTEL_GPU_COPY_PTOP = 10,
};

// gpu_event: lower-rate GPU events, discriminated by kind.
struct otel_gpu_event_rec {
  uint64_t start;
  uint64_t end;
  uint32_t kind; // enum otel_gpu_event_kind
  uint32_t correlation_id;
  uint64_t v1;
  uint64_t v2;
  uint64_t name_ptr;
  uint32_t sample_weight; // events with a correlation_id only
  uint32_t _pad;
};

enum otel_gpu_event_kind {
  OTEL_GPU_EVENT_UVM_HTOD    = 1,  // v1 = bytes migrated host to device
  OTEL_GPU_EVENT_UVM_DTOH    = 2,  // v1 = bytes migrated device to host
  OTEL_GPU_EVENT_UVM_CPU_FLT = 3,  // v1 = CPU page faults
  OTEL_GPU_EVENT_UVM_GPU_FLT = 4,  // v1 = GPU page fault groups
  OTEL_GPU_EVENT_ALLOC       = 5,  // v1 = bytes, v2 = memory kind, correlation_id set
  OTEL_GPU_EVENT_FREE        = 6,  // v1 = bytes, v2 = memory kind, correlation_id set
  OTEL_GPU_EVENT_MEMSET      = 7,  // v1 = bytes, start/end = GPU ns, correlation_id set
  OTEL_GPU_EVENT_STALL       = 8,  // name_ptr = function, v1 = reason string, v2 = samples
  OTEL_GPU_EVENT_COLLECTIVE  = 10, // name_ptr = operation, v1 = bytes, start/end = ns
};

// Memory kinds of allocation events, matching CUpti_ActivityMemoryKind.
enum otel_gpu_memory_kind {
  OTEL_GPU_MEMORY_PAGEABLE       = 1,
  OTEL_GPU_MEMORY_PINNED         = 2,
  OTEL_GPU_MEMORY_DEVICE         = 3,
  OTEL_GPU_MEMORY_ARRAY          = 4,
  OTEL_GPU_MEMORY_MANAGED        = 5,
  OTEL_GPU_MEMORY_DEVICE_STATIC  = 6,
  OTEL_GPU_MEMORY_MANAGED_STATIC = 7,
};

// error: a producer-side failure, for example another CUPTI subscriber holding
// the process. Producers re-emit it periodically: an emission before the
// profiler attached is lost.
struct otel_gpu_error_rec {
  int32_t code; // producer-specific error code
  uint32_t _pad;
  uint64_t msg_ptr; // const char *: static message
};

static_assert(sizeof(struct otel_gpu_launch_rec) == 8, "otel_gpu contract");
static_assert(sizeof(struct otel_gpu_kernel_rec) == 56, "otel_gpu contract");
static_assert(sizeof(struct otel_gpu_mem_rec) == 56, "otel_gpu contract");
static_assert(sizeof(struct otel_gpu_event_rec) == 56, "otel_gpu contract");
static_assert(sizeof(struct otel_gpu_error_rec) == 16, "otel_gpu contract");
