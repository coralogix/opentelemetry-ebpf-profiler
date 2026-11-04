# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

# Reference producer of the otel_gpu_v1 USDT probe consumed by probes/gpu:
#   libotelcupti.so  CUDA injection library (CUDA_INJECTION64_PATH)
#
# CUPTI ships with the CUDA toolkit, not the driver, so the library builds in
# nvidia/cuda devel images:
#   make docker ARCH=amd64 CUDA=13      # one (arch, CUDA major) variant
#   make build-all                      # amd64 and arm64, CUDA 12 and 13
# Ship one library per (arch, CUDA major): it links libcupti.so.<major>.

CC        ?= cc
CUDA_HOME ?= /usr/local/cuda
OUT       ?= libotelcupti.so
CONTRACT  := ../../probes/gpu

# Newer toolkits keep CUPTI under targets/<arch>/ or include/ and lib64/, older
# ones under extras/CUPTI/. Only search CUDA_HOME, not Nsight's bundled copy.
CUPTI_INC := $(dir $(firstword $(shell find $(CUDA_HOME)/ -name cupti.h 2>/dev/null)))
CUPTI_LIB := $(dir $(firstword $(shell find $(CUDA_HOME)/ -name 'libcupti.so*' 2>/dev/null)))

CFLAGS  ?= -O2 -g -Wall -Wextra -fPIC -fvisibility=hidden
LDFLAGS ?= -shared

$(OUT): cupti_producer.c cuda_driver.h sampler.h $(CONTRACT)/otel_gpu_v1_usdt.h
	$(CC) $(CFLAGS) $(LDFLAGS) cupti_producer.c -I$(CUPTI_INC) -I$(CUDA_HOME)/include \
	  -I$(CONTRACT) -L$(CUPTI_LIB) -lcupti -ldl -o $@

sampler_test: test/sampler_test.c sampler.h
	$(CC) -O2 -Wall -Wextra -I. test/sampler_test.c -lm -o $@

# The injection entry point must be exported, the library must not depend on
# libstdc++ (it loads into arbitrary processes, where a second C++ runtime can
# clash with the host's), and the probe needs its semaphore to tell whether a
# profiler is attached.
check: $(OUT) sampler_test
	@./sampler_test
	@nm -D --defined-only $(OUT) | grep -qE ' T InitializeInjection$$' \
	  || { echo "$(OUT): InitializeInjection is not exported"; exit 1; }
	@! readelf -d $(OUT) | grep NEEDED | grep -q stdc++ \
	  || { echo "$(OUT): depends on libstdc++"; exit 1; }
	@$(MAKE) --no-print-directory check-usdt SO=$(OUT)
	@$(MAKE) --no-print-directory check-glibc SO=$(OUT)

# The library loads into applications on older systems, so it must not need a
# C library newer than GLIBC_MAX, the one of RHEL 9 (Ubuntu 22.04 has 2.35).
GLIBC_MAX ?= 2.34
check-glibc:
	@need=$$(objdump -T $(SO) | grep -o 'GLIBC_[0-9.]*' | sed 's/GLIBC_//' | sort -V | tail -1); \
	[ "$$(printf '%s\n%s\n' "$$need" "$(GLIBC_MAX)" | sort -V | tail -1)" = "$(GLIBC_MAX)" ] \
	  || { echo "$(SO): needs glibc $$need, newer than $(GLIBC_MAX)"; exit 1; }; \
	echo "$(SO): needs glibc $$need"

# check-usdt asserts that SO fires otel_gpu_v1:event from a single place, with
# the single pinned register argument of the contract and a semaphore.
check-usdt:
	@notes=$$(readelf -n $(SO) | grep -c 'Provider: '); \
	[ "$$notes" = 1 ] || { echo "$(SO): $$notes USDT probe sites, want 1"; exit 1; }; \
	readelf -n $(SO) | grep -q 'Provider: otel_gpu_v1' \
	  || { echo "$(SO): the probe's provider is not otel_gpu_v1"; exit 1; }; \
	readelf -n $(SO) | grep -q 'Name: event' \
	  || { echo "$(SO): the probe is not named event"; exit 1; }; \
	spec=$$(readelf -n $(SO) | sed -n 's/^ *Arguments: //p'); \
	case "$$spec" in \
	  "8@%rax"|"8@x0") ;; \
	  *) echo "$(SO): argument spec '$$spec' breaks the otel_gpu_v1 contract"; exit 1 ;; \
	esac; \
	readelf -n $(SO) | grep -qE 'Semaphore: 0x0*[1-9a-f]' \
	  || { echo "$(SO): the probe has no semaphore"; exit 1; }; \
	echo "$(SO): otel_gpu_v1:event"

CUDA ?= 13
ARCH ?= amd64
IMG  ?= nvidia/cuda:$(CUDA).0.1-devel-ubuntu24.04

# docker builds one (ARCH, CUDA) variant, libotelcupti.<arch>.cuda<major>.so.
# The repository root is mounted for the contract header.
docker:
	docker run --rm --platform linux/$(ARCH) -v "$(CURDIR)/../..":/src \
	  -w /src/tools/cupti-producer $(IMG) bash -c '\
	  apt-get update -qq && apt-get install -y -qq gcc make systemtap-sdt-dev binutils >/dev/null && \
	  make check OUT=libotelcupti.$(ARCH).cuda$(CUDA).so'

build-all:
	$(MAKE) docker ARCH=amd64 CUDA=12 IMG=nvidia/cuda:12.6.3-devel-ubuntu24.04
	$(MAKE) docker ARCH=amd64 CUDA=13
	$(MAKE) docker ARCH=arm64 CUDA=12 IMG=nvidia/cuda:12.6.3-devel-ubuntu24.04
	$(MAKE) docker ARCH=arm64 CUDA=13

clean:
	rm -f *.so sampler_test

.PHONY: check check-usdt check-glibc docker build-all clean
