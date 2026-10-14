GPU Profiling
=============

# Meta

- **Author(s)**: Mattia Meleleo
- **Start Date**: 2026-10-13
- **Goal End Date**: TBD
- **Primary Reviewers**: [ebpf-profiler-maintainers](https://github.com/orgs/open-telemetry/teams/ebpf-profiler-maintainers)

# Abstract

An opt-in probe that reports GPU time per kernel under the code that launched
it. A library loaded into the workload reports GPU activity through USDT
probes; the profiler captures the host stack at each GPU call and joins it with
the GPU time reported later. Only that library is vendor specific.

# Problem

GPU work is asynchronous: a kernel launch returns before the kernel runs. A CPU
profiler sees the launching thread, not the GPU time it causes, and GPU
utilization metrics do not say which code is responsible.

# Success Criteria

- GPU time is attributed to the host stack that launched the work, in every
  language the profiler unwinds.
- Opt-in, with bounded cost on the workload and the agent, also at thousands of
  kernel launches per second.
- No vendor code in the profiler.
- Existing unwinding, symbolization and export are reused; memory is bounded.

## Scope

Out of scope: the libraries loaded into workloads (they live in their own
repositories), how they are loaded (environment variable, Kubernetes webhook),
and accelerators without a per-call tracing API.

# Proposed Solution

## Architecture

Three parts. Only the first is vendor specific.

1. **Producer library**, in its own repository. It is loaded into the workload
   at startup; for NVIDIA, CUDA loads it from `CUDA_INJECTION64_PATH`. It uses
   the vendor's tracing API (CUPTI) to learn about each GPU call and each
   finished kernel, and reports both through USDT probes. The workload's code
   does not change.
2. **eBPF programs**, in this repository, attached to those probes in each
   process. When the app makes a GPU call, they capture its stack like a CPU
   sample. When GPU work finishes, they pass its record to the agent.
3. **Agent**, in this repository. It matches each GPU record to the stack of
   the call that caused it, adds the GPU time to that stack, and exports the
   result as OTLP profiles.

For one kernel:

1. The app launches a kernel. CUDA gives the call an id.
2. The producer fires `on_launch` with that id, and the profiler captures the
   app thread's stack.
3. The kernel runs. CUPTI later reports its start and end times, with the same
   id.
4. The producer fires `kernel_executed` with the id and the times.
5. The agent finds the stack saved under that id and adds the kernel's GPU time
   to it.

## Contract

The contract is `probes/gpu/otel_gpu_usdt.h`. It is used by producer authors,
not by workloads, which run unchanged. The producer and the profiler are built
and released separately but read the same bytes, so the header fixes the probe
names, the record layouts and how a record is passed.

| Probe | Fired when | Profiler action |
| --- | --- | --- |
| `ready` | Once, when the producer starts | Nothing; attached last, see rule 1 |
| `on_launch` | The app makes a GPU call (kernel launch, memory copy or set, allocation), on the app's thread | Captures the stack, keyed by the call's id |
| `kernel_executed` | A kernel finished on the GPU | Adds its GPU time to its call's stack |
| `gpu_mem` | A memory copy finished | Adds its time and bytes to its call's stack |
| `gpu_event` | Other activity, by kind (memory sets, allocations, ...) | Same, by kind |
| `error` | The producer cannot profile | Logs it |

| Record field | Meaning |
| --- | --- |
| `correlation_id` | The id CUDA gives each call; joins GPU records to the stack |
| `sample_weight` | How many calls this one stands for; 0 counts as 1 |
| `start`, `end` | GPU timestamps, nanoseconds |
| `device_id`, `stream_id`, `graph_id` | Where the work ran; `graph_id` is 0 outside CUDA graphs |
| `name_ptr`, `ranges_ptr` | Kernel name and NVTX ranges, as strings in the producer's memory |

Rules:

1. Producers profile calls only while `OTEL_GPU_ENABLED(ready)` is true. The
   kernel raises it when the profiler attaches to `ready`, which the profiler
   does last, once every other probe works.
2. A call that is not sampled fires nothing: no stack, no GPU records.
3. Strings stay valid while the process runs.
4. `OTEL_GPU_PROBE` passes the record pointer in a fixed register; the
   profiler refuses probes that pass it any other way.
5. Breaking changes use new probe names (`_v2`).

## Sampling

Capturing a stack at every launch costs too much at thousands of launches per
second, so producers sample.

- Each thread samples each kind of call on its own: kernel launches, graph
  launches, memory operations, allocations. Defaults: 200 sampled calls per
  second per thread and kind, 20 for graph launches.
- A call is sampled with probability 1/N and carries weight N, where N follows
  the call rate of the last second. The profiler multiplies times and bytes by
  the weight.

| Approach | Agent load | Totals |
| --- | --- | --- |
| Every call | Unbounded | Exact, until overload drops stacks |
| First call per time slot | Bounded | Biased toward calls after a pause |
| Random 1 in N, weighted (proposed) | Bounded | Unbiased estimate |

## Joining

- GPU records usually arrive before their stack. They wait up to 2 s, then
  count for the process without a stack.
- All kernels of a CUDA graph share one call id and join the same stack.
  Waiting records of one call and kernel are merged into one entry.
- After a process exits, its state is kept 5 s for records in flight; a
  process reusing the PID never matches its stacks.
- Stacks, waiting records and process metadata live in bounded maps.

## Output

Samples are added to each report as aggregated values, without thread or CPU
attributes. A new frame type `gpu` marks the frames the probe creates: the
kernel or operation, then NVTX ranges as `range:` frames, then the host stack.

| Profile | Unit | Step |
| --- | --- | --- |
| `gpu_launches` | count | 3 |
| `gpu_kernel_time` | nanoseconds | 4 |
| `gpu_mem_time`, `gpu_mem_bytes` | nanoseconds, bytes | 4 |
| `gpu_alloc_bytes` | bytes | 6 |
| `gpu_uvm_bytes`, `gpu_uvm_faults` | bytes, count | 6 |
| `gpu_stall_samples` | count | 6 |
| `gpu_collective_time`, `gpu_collective_bytes` | nanoseconds, bytes | 6 |
| `gpu_api_time` | nanoseconds | 6 |

## Changes to the profiler

| Change | Why |
| --- | --- |
| `usdt` exposes each probe's argument string | Check how the record is passed before attaching |
| Frame types made in Go, starting with `gpu`, above the eBPF frame type range | GPU frames are not unwound |
| Probes can add aggregated samples to each report, keeping the metadata of exited processes | GPU time arrives after the launch, sometimes after the process exited |
| Trace handlers can be added while traces are handled | Today this races with the trace loop |
| Probes attach after a process's mappings reach eBPF | Otherwise the first stacks cannot be unwound |

## Vendor support

A vendor needs a way to load a library into unmodified workloads and a tracing
API that calls it on the launching thread with an id the GPU records repeat.
NVIDIA provides both (`CUDA_INJECTION64_PATH` and CUPTI). Other vendors with
both can add a library without profiler changes.

## Plan

| Step | Content |
| --- | --- |
| 1 | The profiler changes above, one PR each |
| 2 | This document |
| 3 | `gpu` probe: contract, attaching, launch stacks |
| 4 | Kernel and memory copy time, join, weights, errors |
| 5 | NVTX ranges |
| 6 | On demand, one PR each: allocations, unified memory, stall samples, collectives, GPU library call time |

# Testing Strategy

- Unit tests for decoding, the join, weights and the sampler.
- Integration tests with a fake producer, a C program firing the probes with
  fixed records, so CI needs no GPU. Totals must match exactly.
- Manual runs on GPU hosts with the NVIDIA library.

The probe is opt-in and creates its eBPF maps only when loaded.

# Open Questions

- Provider name: `otel_gpu`?
- A `gpu` value for `profile.frame.type` in the semantic conventions.
- Where producer libraries live.
- Default sampling rates on hosts with many GPUs or launching threads.

# Decision

TBD.
