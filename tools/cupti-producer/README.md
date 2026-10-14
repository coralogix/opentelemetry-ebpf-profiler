# CUPTI producer for GPU profiling

Reference producer of the `otel_gpu` USDT probes that the `gpu` probe
(`probes/gpu`) consumes. The contract is `probes/gpu/otel_gpu_usdt.h`; any
library that fires those probes works with the profiler.

| Library | Loaded through | Fires |
|---|---|---|
| `libotelcupti.so` | `CUDA_INJECTION64_PATH` (at `cuInit`) | `ready`, `on_launch`, `kernel_executed`, `gpu_mem`, `gpu_event`, `error` |
| `libotelnccl.so` | `NCCL_PROFILER_PLUGIN` (NCCL 2.24 or newer) | `gpu_event` (NCCL operations) |

## Build

CUPTI ships with the CUDA toolkit, so the library builds in `nvidia/cuda`
devel images. Build one variant per architecture and CUDA major version,
because the library links `libcupti.so.<major>`:

```sh
make docker ARCH=amd64 CUDA=13
make build-all   # amd64 and arm64, CUDA 12 and 13
```

`make check` verifies that the injection entry point is exported, that the
library does not depend on `libstdc++`, and that every probe passes its record
in the pinned register (`8@%rax` or `8@x0`).

## Deployment

```sh
CUDA_INJECTION64_PATH=/path/to/libotelcupti.so
LD_LIBRARY_PATH=...                         # must resolve libcupti.so.<major>
NVTX_INJECTION64_PATH=<the workload's libcupti>   # optional, NVTX ranges
NCCL_PROFILER_PLUGIN=/path/to/libotelnccl.so      # optional, NCCL operations
OTEL_GPU_PC_SAMPLING=1                      # optional, stall reasons (Ampere+)
OTEL_GPU_SAMPLE_RATE=200                    # optional, see Sampling
OTEL_GPU_GRAPH_SAMPLE_RATE=20               # optional, see Sampling
```

- Workloads usually ship only the CUDA runtime: stage the toolkit's
  redistributable `libcupti` next to the library.
- For NVTX ranges, `NVTX_INJECTION64_PATH` must point at the `libcupti` the
  library actually loaded. PyTorch bundles its own copy; pointing at another
  one loads a second CUPTI without a subscriber, and NVTX stays silent.
- The workload must start with the variables set.

## Sampling

The library samples call sites, so that launch-heavy workloads do not unwind a
host stack per kernel. Each thread samples kernel launches, memory operations
and allocations at about `OTEL_GPU_SAMPLE_RATE` call sites per second each
(default 200), and CUDA graph launches at `OTEL_GPU_GRAPH_SAMPLE_RATE` (default
20). A sampled call site carries its sample weight, the number of call sites it
stands for, and the profiler scales GPU time and bytes by it. Unsampled call
sites fire no probe at all. `OTEL_GPU_SAMPLE_RATE=0` profiles every call site,
graph launches included unless `OTEL_GPU_GRAPH_SAMPLE_RATE` is set.

Call sites are profiled only while the profiler is attached, which the kernel
signals by raising the semaphore of the `ready` probe. GPU work launched
before the profiler attached is not reported, rather than reported without its
launch stack.

## Limitations

- NVIDIA only; CUDA 12 and 13; Linux amd64 and arm64.
- CUPTI allows one subscriber per process up to CUDA 13.2: next to Kineto,
  Nsight or DCGM the library disables itself and fires `error`, which the
  profiler logs. Errors are re-fired every `OTEL_GPU_ERROR_REEMIT_S` seconds
  (default 60), as one fired before the profiler attached is lost.
- PC sampling needs compute capability 7.0, profiling permission, and in
  practice Ampere or newer: Turing reports no stall reasons.
- Single-rank NCCL collectives do not reach the profiler plugin.

The `test` directory has CUDA workloads exercising NVTX ranges and unified
memory, for manual runs on a GPU host.
