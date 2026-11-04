// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

// The contract between GPU producers and the gpu probe of the OpenTelemetry
// eBPF profiler. A producer is a library loaded into the application, such as
// the CUPTI library in tools/cupti-producer. It reports what the application
// asked the GPU to do, and what the GPU did, as events, and the profiler
// attaches to its USDT probe to read them.
//
// A producer fires a single probe, otel_gpu_v1:event, whose only argument is a
// pointer to a struct otel_gpu_event. The pointer always travels in the same
// register, rax on x86-64 and x0 on arm64, so the profiler never has to work
// out where a compiler put it; it refuses probes that pass it any other way.
// Each place that fires the probe is one more uprobe for the profiler to
// attach, so a producer fires it from one function marked OTEL_GPU_EMITTER,
// which the compiler can neither inline nor copy; the profiler refuses a
// library that fires it from more than one place. A change to the header of
// an event, or to what a kind or a field means, needs a new provider,
// otel_gpu_v2.
//
// An event has a fixed header, which the profiler's eBPF program reads, and a
// payload of up to OTEL_GPU_PAYLOAD_MAX bytes, which only the profiler reads:
// a Payload message in the protobuf wire format,
//
//   message Payload {
//     string gpu = 1;              // the UUID of the GPU that did the work
//     string name = 2;             // the kernel's name, for KERNEL events
//     repeated string ranges = 3;  // names of the work in progress, innermost first
//     uint64 duration_ns = 4;      // how long the GPU worked
//     uint64 bytes = 5;            // how many bytes were copied or filled
//     uint64 runtime = 6;          // the runtime that reported the work
//     uint64 gpu_index = 7;        // the GPU's index, from a DEVICE event
//   }
//
// which otel_gpu_put_string, otel_gpu_put_bytes, otel_gpu_put_whole_bytes and
// otel_gpu_put_uint write. A payload holds at most OTEL_GPU_PAYLOAD_MAX bytes:
// a producer writes the fields of fixed size first, leaves out ranges that do
// not fit whole, and writes the kernel name last, cut short if needed. The
// profiler skips fields it does not know, so new ones need no new provider.
//
// Work that no host call started, such as kernels launched by other kernels,
// is sent with correlation_id 0: the profiler counts it for the process,
// without a stack.
//
// To keep its cost low, the producer may report only some of the
// application's GPU calls. It then sends a LAUNCH event only for the calls it
// reports, and nothing for the work of the other calls. Each reported call
// carries a sample weight, the number of calls it stands for, on its LAUNCH
// event and on the events of its work, and the profiler multiplies the call's
// GPU work by it.
//
// The producer should report only while the profiler is attached, because work
// started earlier would arrive without the stack of its call. Defining
// OTEL_GPU_SEMAPHORE before including this file, in one source file, gives the
// probe a counter that the kernel raises while a tool is attached to it, and
// the producer reports while OTEL_GPU_ENABLED() is true.
#pragma once

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef OTEL_GPU_SEMAPHORE
  #define _SDT_HAS_SEMAPHORES 1
// The kernel can only update counters that live in the file, hence .probes.
__attribute__((
  section(".probes"), visibility("hidden"))) volatile unsigned short otel_gpu_v1_event_semaphore;
  #define OTEL_GPU_ENABLED() (otel_gpu_v1_event_semaphore != 0)
#endif

#include <sys/sdt.h>

#if defined(__x86_64__)
  #define OTEL_GPU_EVENT_REGISTER "rax"
#elif defined(__aarch64__)
  #define OTEL_GPU_EVENT_REGISTER "x0"
#else
  #error "otel_gpu_v1 USDT probes support x86-64 and arm64 only"
#endif

// OTEL_GPU_EMITTER marks the one function that fires the probe: the compiler
// must neither inline it nor copy it, as each copy is one more probe site.
#if defined(__clang__)
  #define OTEL_GPU_EMITTER __attribute__((noinline))
#else
  #define OTEL_GPU_EMITTER __attribute__((noipa))
#endif

// OTEL_GPU_EVENT fires the probe with the event `ev`.
#define OTEL_GPU_EVENT(ev)                                                                         \
  do {                                                                                             \
    register const void *otel_gpu_event_ asm(OTEL_GPU_EVENT_REGISTER) = (ev);                      \
    DTRACE_PROBE1(otel_gpu_v1, event, otel_gpu_event_);                                            \
  } while (0)

// What happened. The profiler ignores events of kinds it does not know.
enum otel_gpu_kind {
  // The application asked the GPU to do something. Sent on the application's
  // own thread: the profiler records the thread's stack, so it can later show
  // which code caused the work. Has no payload.
  OTEL_GPU_LAUNCH    = 1,
  // The GPU ran a kernel, a function that runs on the GPU. Its name is the
  // kernel's name, as the compiler wrote it.
  OTEL_GPU_KERNEL    = 2,
  // The GPU filled memory with a value.
  OTEL_GPU_FILL      = 3,
  // The GPU copied memory, from the CPU's memory (H, host) or the GPU's (D,
  // device) to either. Copies from or to CUDA arrays count as the GPU's.
  OTEL_GPU_COPY_HTOD = 4,
  OTEL_GPU_COPY_DTOH = 5,
  OTEL_GPU_COPY_DTOD = 6,
  OTEL_GPU_COPY_HTOH = 7,
  // A GPU of the process: its gpu field is the UUID that the gpu_index field
  // stands for in the events of the process's work. The producer sends it
  // before the first work on that GPU since tracing started, and again from
  // time to time, in case it was lost.
  OTEL_GPU_DEVICE    = 8,
};

// The runtime that reported the work.
enum otel_gpu_runtime {
  OTEL_GPU_RUNTIME_CUDA = 1,
  OTEL_GPU_RUNTIME_ROCM = 2,
};

// The fields of a payload.
enum otel_gpu_field {
  OTEL_GPU_FIELD_GPU         = 1,
  OTEL_GPU_FIELD_NAME        = 2,
  OTEL_GPU_FIELD_RANGES      = 3,
  OTEL_GPU_FIELD_DURATION_NS = 4,
  OTEL_GPU_FIELD_BYTES       = 5,
  OTEL_GPU_FIELD_RUNTIME     = 6,
  OTEL_GPU_FIELD_GPU_INDEX   = 7,
};

// The largest payload the profiler accepts; it drops events with a larger one.
enum { OTEL_GPU_PAYLOAD_MAX = 4096 };

struct otel_gpu_event {
  // The call id. The work the call causes is reported later with the same id.
  // The profiler tells calls apart by the low 32 bits, so those must not
  // repeat within a few seconds.
  uint64_t correlation_id;
  // How many calls this one stands for. 0 means 1.
  uint32_t sample_weight;
  // What happened, from enum otel_gpu_kind.
  uint16_t kind;
  // How many bytes of payload follow.
  uint16_t payload_size;
  uint8_t payload[OTEL_GPU_PAYLOAD_MAX];
};

static_assert(offsetof(struct otel_gpu_event, payload) == 16, "otel_gpu_v1 contract");

// otel_gpu_varint_ writes v as a protobuf varint at p and returns its length.
static inline size_t otel_gpu_varint_(uint8_t *p, uint64_t v)
{
  size_t n = 0;
  while (v >= 0x80) {
    p[n++] = (uint8_t)(v | 0x80);
    v >>= 7;
  }
  p[n++] = (uint8_t)v;
  return n;
}

// otel_gpu_put_uint appends a number field to the payload, unless it is 0 or
// does not fit.
static inline void
otel_gpu_put_uint(struct otel_gpu_event *ev, enum otel_gpu_field field, uint64_t value)
{
  uint8_t field_value[11];
  const size_t key_size = otel_gpu_varint_(field_value, (uint64_t)field << 3);
  const size_t size     = key_size + otel_gpu_varint_(field_value + key_size, value);
  if (value == 0 || ev->payload_size + size > OTEL_GPU_PAYLOAD_MAX) {
    return;
  }
  memcpy(ev->payload + ev->payload_size, field_value, size);
  ev->payload_size += (uint16_t)size;
}

// otel_gpu_put_bytes appends a string field of len bytes to the payload, cut
// short if it does not fit whole. Empty strings are left out.
static inline void otel_gpu_put_bytes(
  struct otel_gpu_event *ev, enum otel_gpu_field field, const char *data, size_t len)
{
  // A key and a length of at most OTEL_GPU_PAYLOAD_MAX take up to 3 bytes.
  enum { prefix_max = 3 };
  if (len == 0 || ev->payload_size + prefix_max >= OTEL_GPU_PAYLOAD_MAX) {
    return;
  }
  const size_t space = OTEL_GPU_PAYLOAD_MAX - ev->payload_size - prefix_max;
  if (len > space) {
    len = space;
  }
  uint8_t *p = ev->payload + ev->payload_size;
  size_t n   = otel_gpu_varint_(p, (uint64_t)field << 3 | 2);
  n += otel_gpu_varint_(p + n, len);
  memcpy(p + n, data, len);
  ev->payload_size += (uint16_t)(n + len);
}

// otel_gpu_put_whole_bytes appends a string field of len bytes to the payload
// only if it fits whole, and reports whether it did. Empty strings are left
// out.
static inline int otel_gpu_put_whole_bytes(
  struct otel_gpu_event *ev, enum otel_gpu_field field, const char *data, size_t len)
{
  // A key and a length of at most OTEL_GPU_PAYLOAD_MAX take up to 3 bytes.
  enum { prefix_max = 3 };
  if (len == 0 || ev->payload_size + prefix_max + len > OTEL_GPU_PAYLOAD_MAX) {
    return 0;
  }
  otel_gpu_put_bytes(ev, field, data, len);
  return 1;
}

// otel_gpu_put_string appends a string field to the payload, cut short if it
// does not fit whole. Empty and missing strings are left out.
static inline void
otel_gpu_put_string(struct otel_gpu_event *ev, enum otel_gpu_field field, const char *s)
{
  if (s) {
    otel_gpu_put_bytes(ev, field, s, strlen(s));
  }
}
