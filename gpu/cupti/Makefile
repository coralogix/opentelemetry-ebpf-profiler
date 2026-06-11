# SPDX-License-Identifier: Apache-2.0
# libotelcupti.so (CUDA injection shim) + libotelnccl.so (NCCL profiler
# plugin), plain C, plain make.
#
# CUPTI ships with the CUDA toolkit (NOT the driver), so shims build inside
# nvidia/cuda:<maj>-devel images (+ systemtap-sdt-dev for sys/sdt.h):
#   make                                  # native build (needs CUDA toolkit)
#   make docker ARCH=amd64 CUDA=13        # one (arch, CUDA major) variant
#   make docker ARCH=arm64 CUDA=12 IMG=nvidia/cuda:12.6.3-devel-ubuntu24.04
#   make build-all                        # all four shim variants
# Ship one shim per (arch, CUDA major); the agent's injection setup picks the
# major matching the workload's libcudart.so.<maj>. The NCCL plugin is
# CUDA-independent and built once per arch alongside.

CC        ?= cc
CUDA_HOME ?= /usr/local/cuda
OUT       ?= libotelcupti.so

# Locate CUPTI in the toolkit (newer CUDA: targets/<arch>/{include,lib} or
# include/ + lib64/; older: extras/CUPTI/...). Search only under CUDA_HOME so we
# don't pick up Nsight's bundled copy.
CUPTI_INC := $(dir $(firstword $(shell find $(CUDA_HOME)/ -name cupti.h 2>/dev/null)))
CUPTI_LIB := $(dir $(firstword $(shell find $(CUDA_HOME)/ -name 'libcupti.so*' 2>/dev/null)))

CFLAGS  ?= -O2 -g -Wall -Wextra -fPIC -fvisibility=hidden
LDFLAGS ?= -shared

$(OUT): shim.c usdt_probes.h
	$(CC) $(CFLAGS) $(LDFLAGS) shim.c -I$(CUPTI_INC) -I. -L$(CUPTI_LIB) -lcupti -o $@

# NCCL profiler plugin: no CUDA/NCCL link, just the vendored ABI headers.
NCCL_OUT ?= libotelnccl.so
$(NCCL_OUT): ncclprofiler.c usdt_probes.h nccl/profiler_v2.h nccl/profiler_v2_compat.h
	$(CC) $(CFLAGS) $(LDFLAGS) ncclprofiler.c -I. -o $@

# Injection entry DEFINED and exported + no direct libstdc++ dependency
# (the shim loads into arbitrary processes; a second C++ runtime can clash
# with the host's) + the USDT contract via check-usdt.
check: $(OUT)
	@nm -D --defined-only $(OUT) | grep -qE ' T InitializeInjection$$' \
	  && echo "InitializeInjection exported ✓" \
	  || { echo "InitializeInjection not a defined exported function"; exit 1; }
	@! readelf -d $(OUT) | grep NEEDED | grep -q stdc++ && echo "no libstdc++ ✓"
	@$(MAKE) --no-print-directory check-usdt SO=$(OUT) PROBES='$(SHIM_PROBES)'

SHIM_PROBES := error gpu_event gpu_mem kernel_executed on_launch

# check-usdt asserts SO's USDT contract: the probe set is EXACTLY $(PROBES)
# (sorted; a new probe must be enrolled to be spec-checked) and every arg
# spec is the pinned single register (8@%rax / 8@x0) — anything else silently
# breaks the eBPF reads. No build dependency on purpose: CI also runs this
# against the COMMITTED blobs.
check-usdt:
	@found=$$(readelf -n $(SO) | sed -n 's/^ *Name: //p' | sort -u | tr '\n' ' '); \
	[ "$$found" = "$(PROBES) " ] || { echo "$(SO): probe set '$$found' != '$(PROBES)'"; exit 1; }; \
	for p in $(PROBES); do \
	  specs=$$(readelf -n $(SO) | awk -v p="$$p" '/^ *Name: /{n=$$2} /^ *Arguments: /{if (n==p) print $$2}'); \
	  [ -n "$$specs" ] || { echo "USDT $$p: no .note.stapsdt found in $(SO)"; exit 1; }; \
	  for spec in $$specs; do \
	    case "$$spec" in \
	      "8@%rax"|"8@x0") ;; \
	      *) echo "USDT $$p arg spec '$$spec' is NOT the expected single-register form"; exit 1 ;; \
	    esac; \
	  done; \
	  echo "$(SO) USDT $$p arg spec(s): $$specs ✓" | tr '\n' ' '; echo ""; \
	done

# ─── Cross-build matrix: {amd64, arm64} × {CUDA 12, 13} ───────────────────────
# Production ships a shim per (arch, CUDA major). NVIDIA cuda devel images are
# multi-arch (x86_64 + sbsa/arm64) under the same tag; --platform selects.
CUDA ?= 13
ARCH ?= amd64
IMG  ?= nvidia/cuda:$(CUDA).0.1-devel-ubuntu24.04

# Build one (ARCH, CUDA) variant in a container → libotelcupti.<arch>.cuda<maj>.so
# (+ the CUDA-independent NCCL plugin, once per arch).
docker:
	docker run --rm --platform linux/$(ARCH) -v "$(CURDIR)":/src -w /src $(IMG) bash -c '\
	  apt-get update -qq && apt-get install -y -qq gcc systemtap-sdt-dev binutils >/dev/null && \
	  make check OUT=libotelcupti.$(ARCH).cuda$(CUDA).so && \
	  make libotelnccl.$(ARCH).so NCCL_OUT=libotelnccl.$(ARCH).so && \
	  make check-usdt SO=libotelnccl.$(ARCH).so PROBES=gpu_event'

# All four production artifacts.
build-all:
	$(MAKE) docker ARCH=amd64 CUDA=12
	$(MAKE) docker ARCH=amd64 CUDA=13
	$(MAKE) docker ARCH=arm64 CUDA=12
	$(MAKE) docker ARCH=arm64 CUDA=13

clean:
	rm -f $(OUT) libotelcupti.*.so

.PHONY: check check-usdt docker build-all clean
