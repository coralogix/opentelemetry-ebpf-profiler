# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

# Reference producer of the otel_gpu USDT probes consumed by probes/gpu:
#   libotelcupti.so  CUDA injection library (CUDA_INJECTION64_PATH)
#   libotelnccl.so   NCCL profiler plugin (NCCL_PROFILER_PLUGIN)
#
# CUPTI ships with the CUDA toolkit, not the driver, so the library builds in
# nvidia/cuda devel images:
#   make docker ARCH=amd64 CUDA=13      # one (arch, CUDA major) variant
#   make build-all                      # amd64 and arm64, CUDA 12 and 13
# Ship one library per (arch, CUDA major): it links libcupti.so.<major>. The
# NCCL plugin does not depend on CUDA and builds once per arch.

CC        ?= cc
CUDA_HOME ?= /usr/local/cuda
OUT       ?= libotelcupti.so
NCCL_OUT  ?= libotelnccl.so
CONTRACT  := ../../probes/gpu

# Newer toolkits keep CUPTI under targets/<arch>/ or include/ and lib64/, older
# ones under extras/CUPTI/. Only search CUDA_HOME, not Nsight's bundled copy.
CUPTI_INC := $(dir $(firstword $(shell find $(CUDA_HOME)/ -name cupti.h 2>/dev/null)))
CUPTI_LIB := $(dir $(firstword $(shell find $(CUDA_HOME)/ -name 'libcupti.so*' 2>/dev/null)))

CFLAGS  ?= -O2 -g -Wall -Wextra -fPIC -fvisibility=hidden
LDFLAGS ?= -shared

$(OUT): cupti_producer.c sampler.h $(CONTRACT)/otel_gpu_usdt.h
	$(CC) $(CFLAGS) $(LDFLAGS) cupti_producer.c -I$(CUPTI_INC) -I$(CONTRACT) \
	  -L$(CUPTI_LIB) -lcupti -o $@

$(NCCL_OUT): nccl_profiler.c $(CONTRACT)/otel_gpu_usdt.h nccl/*.h
	$(CC) $(CFLAGS) $(LDFLAGS) nccl_profiler.c -I. -I$(CONTRACT) -o $@

sampler_test: test/sampler_test.c sampler.h
	$(CC) -O2 -Wall -Wextra -I. test/sampler_test.c -lm -o $@

CUPTI_PROBES := error gpu_event gpu_mem kernel_executed on_launch ready

# The injection entry point must be exported, the library must not depend on
# libstdc++ (it loads into arbitrary processes, where a second C++ runtime can
# clash with the host's), and ready needs its semaphore to tell whether a
# profiler is attached.
check: $(OUT) $(NCCL_OUT) sampler_test
	@./sampler_test
	@nm -D --defined-only $(OUT) | grep -qE ' T InitializeInjection$$' \
	  || { echo "$(OUT): InitializeInjection is not exported"; exit 1; }
	@! readelf -d $(OUT) | grep NEEDED | grep -q stdc++ \
	  || { echo "$(OUT): depends on libstdc++"; exit 1; }
	@$(MAKE) --no-print-directory check-usdt SO=$(OUT) PROBES='$(CUPTI_PROBES)'
	@readelf -n $(OUT) | grep -A2 'Name: ready' | grep -qE 'Semaphore: 0x0*[1-9a-f]' \
	  || { echo "$(OUT): ready has no semaphore"; exit 1; }
	@$(MAKE) --no-print-directory check-usdt SO=$(NCCL_OUT) PROBES=gpu_event

# check-usdt asserts that SO fires exactly PROBES, each with the single pinned
# register argument of the otel_gpu contract.
check-usdt:
	@found=$$(readelf -n $(SO) | sed -n 's/^ *Name: //p' | sort -u | tr '\n' ' '); \
	[ "$$found" = "$(PROBES) " ] || { echo "$(SO): probes '$$found' != '$(PROBES)'"; exit 1; }; \
	for spec in $$(readelf -n $(SO) | sed -n 's/^ *Arguments: //p'); do \
	  case "$$spec" in \
	    "8@%rax"|"8@x0") ;; \
	    *) echo "$(SO): argument spec '$$spec' breaks the otel_gpu contract"; exit 1 ;; \
	  esac; \
	done; \
	echo "$(SO): otel_gpu probes $(PROBES)"

CUDA ?= 13
ARCH ?= amd64
IMG  ?= nvidia/cuda:$(CUDA).0.1-devel-ubuntu24.04

# docker builds one (ARCH, CUDA) variant: libotelcupti.<arch>.cuda<major>.so and
# libotelnccl.<arch>.so. The repository root is mounted for the contract header.
docker:
	docker run --rm --platform linux/$(ARCH) -v "$(CURDIR)/../..":/src \
	  -w /src/tools/cupti-producer $(IMG) bash -c '\
	  apt-get update -qq && apt-get install -y -qq gcc make systemtap-sdt-dev binutils >/dev/null && \
	  make check OUT=libotelcupti.$(ARCH).cuda$(CUDA).so NCCL_OUT=libotelnccl.$(ARCH).so'

build-all:
	$(MAKE) docker ARCH=amd64 CUDA=12 IMG=nvidia/cuda:12.6.3-devel-ubuntu24.04
	$(MAKE) docker ARCH=amd64 CUDA=13
	$(MAKE) docker ARCH=arm64 CUDA=12 IMG=nvidia/cuda:12.6.3-devel-ubuntu24.04
	$(MAKE) docker ARCH=arm64 CUDA=13

clean:
	rm -f *.so sampler_test

.PHONY: check check-usdt docker build-all clean
