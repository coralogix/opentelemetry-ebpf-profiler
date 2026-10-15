# ROCm producer for GPU profiling

Reference producer of the `otel_gpu` USDT probes for AMD GPUs, the ROCm
counterpart of `tools/cupti-producer`. The profiler side is the same `gpu`
probe; the contract is `probes/gpu/otel_gpu_usdt.h`.

| Library | Loaded through | Fires |
|---|---|---|
| `libotelrocm.so` | `ROCP_TOOL_LIBRARIES` (rocprofiler-sdk, at HIP start) | `ready`, `on_launch`, `kernel_executed`, `gpu_mem`, `gpu_event`, `error` |

It uses rocprofiler-sdk, ROCm's tracing interface: HIP calls fire `on_launch`
on the calling thread, and the kernel dispatch, memory copy and memory
allocation records that rocprofiler-sdk reports later carry the same
correlation id. ROCTx ranges (`roctxRangePush`) become range frames, like NVTX
ranges with CUDA.

## Build

Needs ROCm 6.4 or newer, which ships rocprofiler-sdk:

```sh
make check            # ROCm in /opt/rocm, or set ROCM_PATH
make docker           # in the rocm/dev-ubuntu-24.04 image, amd64 only
```

`make check` verifies that `rocprofiler_configure` is exported, that the
library does not depend on `libstdc++`, and that every probe passes its record
in the pinned register.

## Deployment

```sh
ROCP_TOOL_LIBRARIES=/path/to/libotelrocm.so
OTEL_GPU_SAMPLE_RATE=200          # optional, as for the CUDA producer
OTEL_GPU_GRAPH_SAMPLE_RATE=20     # optional
```

Sampling and the `ready` gate work as in the CUDA producer.

## Differences from the CUDA producer

- Correlation ids are 64-bit in rocprofiler-sdk; records carry the low 32 bits.
- HIP does not report graph ids: kernels of a `hipGraphLaunch` carry graph id 1.
- No unified memory, stall samples or collectives yet.
