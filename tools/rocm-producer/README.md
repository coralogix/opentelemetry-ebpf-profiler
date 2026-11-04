# ROCm producer for GPU profiling

Reference producer of the `otel_gpu_v1:event` USDT probe for AMD GPUs, the
ROCm counterpart of `tools/cupti-producer`. The profiler side is the same `gpu`
probe; the contract is `probes/gpu/otel_gpu_v1_usdt.h`.

| Library | Loaded through | Sends |
|---|---|---|
| `libotelrocm.so` | `ROCP_TOOL_LIBRARIES` (rocprofiler-sdk, at HIP start) | `LAUNCH`, `KERNEL` and `COPY` events |

It uses rocprofiler-sdk, ROCm's tracing interface: HIP calls that launch
kernels or graphs, or copy or fill memory, send `LAUNCH` events on the calling
thread, and the kernel dispatch and memory copy records that rocprofiler-sdk
reports later carry the same correlation id. ROCTx ranges (`roctxRangePushA`)
become range frames, like NVTX ranges with CUDA.

It has not been tested on AMD GPUs yet: it builds with ROCm 7.0 and newer.

## Build

Needs ROCm 7.0 or newer, whose rocprofiler-sdk reports full GPU UUIDs:

```sh
make check            # ROCm in /opt/rocm, or set ROCM_PATH
make docker           # in the rocm/dev-ubuntu-24.04 image, amd64 only
```

`make check` verifies that `rocprofiler_configure` is exported, that the
library does not depend on `libstdc++`, and that it fires the probe from a
single place, passing the event in the pinned register and with a semaphore.

## Deployment

```sh
ROCP_TOOL_LIBRARIES=/path/to/libotelrocm.so
OTEL_GPU_SAMPLE_RATE=200          # optional, as for the CUDA producer
OTEL_GPU_GRAPH_SAMPLE_RATE=20     # optional
```

Sampling, and recording GPU work only while the profiler is attached, work as
in the CUDA producer.

## Differences from the CUDA producer

- No `FILL` events: rocprofiler-sdk reports memory fills as the kernels that do
  them.
- GPUs are named by their UUID alone, without the `GPU-` prefix of NVIDIA's.
- Copies between GPUs are reported, as copies from the GPU's memory to the
  GPU's.
- Only ranges pushed with `roctxRangePushA` are named.
