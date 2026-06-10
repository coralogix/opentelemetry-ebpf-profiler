# GPU profiling (`gpu/`)

Opt-in (`--gpu`), NVIDIA-only. Captures GPU activity with real hardware
timing and attributes it to the host call stack that caused it, exported as
OTLP profiles through the same reporter pipeline as CPU samples.

## Sample types

All are dynamic trace origins; a value-carrying leaf-first stack per sample.

| Type | Unit | Source | Needs shim |
|---|---|---|---|
| `gpu_kernel_time` | ns | CUPTI Activity (per-kernel HW timestamps) | yes |
| `gpu_mem_time` / `gpu_mem_bytes` | ns / bytes | CUPTI memcpy + memset activity | yes |
| `gpu_alloc_bytes` | bytes | CUPTI MEMORY2 (cudaMalloc/Free per memory kind) | yes |
| `gpu_uvm_bytes` / `gpu_uvm_faults` | bytes / count | CUPTI unified-memory counters (migrations, page faults) | yes |
| `gpu_stall_samples` | count | CUPTI PC sampling (per-function stall reasons) | yes + `OTELCUPTI_PC=1` |
| `gpu_api_time` | ns | cuDNN/cuBLAS API spans (uprobe/uretprobe pairs) | yes (attach piggybacks on the shim attach) |
| `gpu_nccl_time` / `gpu_nccl_bytes` | ns / bytes | NCCL profiler plugin (collective/p2p ops) | plugin |
| `gpu_busy_time` | ns | NVML per-process utilization (`nvidia-smi pmon`) | no |

Kernel, memcpy, memset and alloc samples carry the host launch stack
(collapsed to one frame per shared object — stripped CUDA stacks have no
symbols) plus the NVTX range nesting active at the call, e.g.
`memcpy:HtoD → nvtx:aten::cudnn_batch_norm → … → nvtx:train_step → libcudart → python3.11`.

## Architecture

```
 CUDA workload
   ├─ libotelcupti.so   injected via CUDA_INJECTION64_PATH (loaded at cuInit)
   │    CUPTI callbacks: launch/memcpy/memset/malloc call sites
   │      → USDT otelcupti:on_launch(&rec)        (correlation id)
   │    CUPTI Activity: kernels, memcpys, memsets, allocs, UVM counters
   │      → USDT kernel_executed / gpu_mem / gpu_event
   │    CUPTI PC sampling: stall-reason histograms → gpu_event
   │    errors (e.g. another CUPTI subscriber) → USDT error, re-emitted
   └─ libotelnccl.so    NCCL profiler plugin (NCCL_PROFILER_PLUGIN)
        collective/p2p op spans → USDT gpu_event

 support/ebpf/gpu_cupti.ebpf.c (per-PID uprobes on the USDT sites)
   on_launch → collect_trace(TRACE_GPU): host stack keyed by correlation id
   others    → ringbufs (cupti_events / cupti_mem_events / cupti_misc_events
               / cupti_errors)
   cuDNN/cuBLAS API spans: uprobe+uretprobe pairs, cookie = symbol index

 gpu/cupti (Go)
   Source   attaches per PID (PID-event hook + 2s rescan; libraries dlopen'd
            after cuInit are picked up), drains the ringbufs
   Matcher  joins timing to host stacks on (PID, correlation id), aggregates
            per (process, NVTX, leaf, stack), flushes deltas every 5s
   reporter → OTLP profiles
```

Every USDT probe passes a single struct pointer pinned to a fixed register
(`8@%rax` amd64 / `8@x0` arm64) — see `usdt_probes.h`. The agent verifies the
spec at attach time and refuses mismatched shim builds.

Host-stack capture is sampled (a full unwind per launch); kernel timing is
complete. Correlation ids are per-process counters, hence the (PID, corr)
join key.

## Deployment

The shim is injected per workload:

```
CUDA_INJECTION64_PATH=/path/to/libotelcupti.so   # loaded by CUDA at cuInit
LD_LIBRARY_PATH=…                                # must resolve libcupti.so.<maj>
NVTX_INJECTION64_PATH=<workload's own libcupti>  # NVTX ranges (see below)
NCCL_PROFILER_PLUGIN=/path/to/libotelnccl.so     # optional, NCCL >= 2.24
```

- The shim links `libcupti.so.<major>`, which workloads do not ship (they
  carry only the CUDA runtime): stage the toolkit's redistributable libcupti
  next to the shim. Use the toolkit copy — Nsight's does not export
  `InitializeInjectionNvtx2`.
- NVTX: the shim's `NEEDED libcupti` resolves to the workload's own copy if
  one is already loaded (PyTorch bundles one), so `NVTX_INJECTION64_PATH`
  must point at **that** instance — pointing at our bundled copy loads a
  second CUPTI with no subscriber and NVTX stays silent.
- The workload must (re)start to pick up the injection.

Run the agent with `--gpu`. Processes without the shim still get
`gpu_busy_time` (NVML); all other sample types, including `gpu_api_time`,
require the shim (probe attach is keyed on it).

## Build

```
make -C gpu/cupti docker ARCH=amd64 CUDA=13   # shim variant + NCCL plugin
make -C gpu/cupti build-all                   # all four shim variants
```

Plain C, gcc, no C++ runtime (the shim loads into arbitrary processes).
`make check` asserts: injection entry exported, USDT notes present, no
libstdc++, and every probe's arg spec is the pinned single register. CI
(`.github/workflows/gpu.yml`) builds the matrix and runs the Go tests.

The eBPF consumers compile into the main tracer blob (`support/ebpf`).

## Requirements & limitations

- NVIDIA only; CUDA 12.x/13.x; Linux amd64/arm64.
- ~300 MB GPU memory per GPU while CUPTI is active; `CAP_SYS_PTRACE` to read
  names from target processes.
- CUPTI allows one subscriber per process on CUDA ≤ 13.2 / driver < r610:
  with Kineto/Nsight/DCGM already subscribed the shim disables itself and the
  conflict surfaces as an agent log warning (the `error` probe is re-emitted
  periodically — the cuInit-time emission races the USDT attach).
- PC sampling needs CC ≥ 7.0 and profiling permission
  (`RmProfilingAdminOnly=0` or root); enables cleanly but sample retrieval is
  still being validated — Turing (T4) enumerates zero stall reasons, so it
  needs Ampere+. Source-line correlation would additionally need `-lineinfo`
  cubins. Opt-in: set `OTELCUPTI_PC=1` in the workload (off by default).
- PM sampling (Hopper+) is not implemented; `OTELCUPTI_PM=1` surfaces that as
  an error probe.
- NCCL plugin: single-rank collectives short-circuit inside NCCL and emit no
  events; multi-GPU validation pending.
- `gpu_api_time` covers a curated symbol set (`gpu/cupti/apisymbols.go`);
  cuDNN 9 routes most work through `cudnnBackendExecute`, which is included.

## Tests

`go test ./gpu/cupti/` (Linux only — run in a container on other hosts).
Covers the matcher join/delta/prune logic, wire-format size pins and
demangling. `gpu/cupti/test/` has standalone CUDA workloads (NVTX, UVM) and a
USDT producer for attach validation without a GPU.
