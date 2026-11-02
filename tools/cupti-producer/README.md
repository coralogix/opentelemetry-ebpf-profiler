# CUPTI producer for GPU profiling

Reference producer of the `otel_gpu_v1:event` USDT probe that the `gpu` probe
(`probes/gpu`) consumes. The contract is `probes/gpu/otel_gpu_v1_usdt.h`; any
library that fires that probe works with the profiler.

| Library | Loaded through | Sends |
|---|---|---|
| `libotelcupti.so` | `CUDA_INJECTION64_PATH` (at `cuInit`) | `LAUNCH`, `KERNEL`, `COPY` and `FILL` events |

## Build

CUPTI ships with the CUDA toolkit, so the library builds in `nvidia/cuda`
devel images. Build one variant per architecture and CUDA major version,
because the library links `libcupti.so.<major>`:

```sh
make docker ARCH=amd64 CUDA=13
make build-all   # amd64 and arm64, CUDA 12 and 13
```

`make check` verifies that the injection entry point is exported, that the
library does not depend on `libstdc++`, and that it fires the probe from a
single place, passing the event in the pinned register (`8@%rax` or `8@x0`)
and with a semaphore.

## Deployment

```sh
CUDA_INJECTION64_PATH=/path/to/libotelcupti.so
LD_LIBRARY_PATH=...                         # must resolve libcupti.so.<major>
NVTX_INJECTION64_PATH=<the workload's libcupti>   # optional, NVTX ranges
OTEL_GPU_SAMPLE_RATE=200                    # optional, see Sampling
OTEL_GPU_GRAPH_SAMPLE_RATE=20               # optional, see Sampling
OTEL_GPU_DEBUG_TOTALS=1                     # optional, see Debug totals
```

- Workloads usually ship only the CUDA runtime: stage the toolkit's
  redistributable `libcupti` next to the library.
- For NVTX ranges, `NVTX_INJECTION64_PATH` must point at the `libcupti` the
  library actually loaded. PyTorch bundles its own copy; pointing at another
  one loads a second CUPTI without a subscriber, and NVTX stays silent.
- The workload must start with the variables set.

## Sampling

The library tracks the CUDA runtime and driver calls that launch kernels or
graphs, or copy or fill memory. It finds them by name when it starts, so that
every variant is tracked, also those of later CUDA versions.

The library samples the tracked calls, so that launch-heavy workloads do not
unwind a host stack per kernel. Each thread samples kernel launches and memory
operations at about `OTEL_GPU_SAMPLE_RATE` calls per second each (default 200),
and CUDA graph launches at `OTEL_GPU_GRAPH_SAMPLE_RATE` (default 20). A sampled
call's `LAUNCH` event and the events of its GPU work carry its sample weight,
the number of calls it stands for, and the profiler scales the call's GPU time
and bytes by it. Unsampled calls send no event at all. `OTEL_GPU_SAMPLE_RATE=0`
profiles every call, graph launches included unless
`OTEL_GPU_GRAPH_SAMPLE_RATE` is set.

The library traces only while the profiler is attached, which the kernel
signals by raising the semaphore of the probe: a thread of the library checks
it twice a second and turns CUPTI activity tracing on when it rises and off
when it drops. The same thread asks CUPTI for the records of finished work, so
that they reach the profiler within a second, and the application's threads
never wait for either. Without a profiler, the library costs a check of the
semaphore at each tracked CUDA call, and keeps the NVTX ranges of each thread.
GPU work launched before tracing started is not reported, rather than reported
without its launch stack.

## Debug totals

With `OTEL_GPU_DEBUG_TOTALS=1`, the library adds up the GPU work that started
while it traced, sampled or not, and what its samples stand for. It prints both
to stderr as one line,
`otelcupti: debug totals: pid=… exact_kernel_ns=… estimated_kernel_ns=… largest_payload=…`,
with the kernel time, the time and bytes of memory operations, and the largest
payload it sent: every 10 seconds while it traces, and when tracing stops or
the process exits. A profile of the same period adds up to the estimated
totals, and estimates the exact ones. Tracing stops within half a second of the
profiler detaching; work that finishes in between counts here but does not
reach the profiler.

## Limitations

- NVIDIA only; CUDA 12 and 13; Linux amd64 and arm64, glibc 2.34 or newer.
- Copies between GPUs are not reported: CUPTI reports them as records of
  another kind, which the library does not turn on.
- CUPTI serves one tool per process, the first to subscribe, and the library
  subscribes as soon as CUDA starts. Next to it, PyTorch's profiler records no
  GPU activity and logs `CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED`; a tool
  that subscribed before CUDA started keeps CUPTI, and the library disables
  itself and logs why to stderr. Tested with CUDA 12.8 and 13.0.
- NVTX ranges of all domains share one stack per thread, so ranges of
  different domains that are not nested inside each other get mislabeled.

The `test` directory has CUDA workloads exercising every kind of event and
NVTX ranges, for manual runs on a GPU host.
