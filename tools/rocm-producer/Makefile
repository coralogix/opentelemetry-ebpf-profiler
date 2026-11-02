# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

# Reference producer of the otel_gpu_v1 USDT probe for AMD GPUs:
#   libotelrocm.so  rocprofiler-sdk tool library (ROCP_TOOL_LIBRARIES)
#
# Builds on a host or image with ROCm 7.0 or newer (rocprofiler-sdk):
#   make check                # with ROCm in /opt/rocm
#   make docker               # in the rocm/dev-ubuntu-24.04 image (amd64)

CXX       ?= g++
CC        ?= gcc
ROCM_PATH ?= /opt/rocm
OUT       ?= libotelrocm.so
CONTRACT  := ../../probes/gpu
SAMPLER   := ../cupti-producer

# C++ only for the rocprofiler-sdk headers: no exceptions, no RTTI, and the
# link uses the C driver, so the library never depends on libstdc++. Only
# hipcc selects the HIP platform itself.
CXXFLAGS ?= -O2 -g -Wall -Wextra -std=gnu++17 -fPIC -fvisibility=hidden -fno-exceptions -fno-rtti
LDFLAGS  ?= -shared -Wl,-z,defs

$(OUT): rocm_producer.cpp $(SAMPLER)/sampler.h $(CONTRACT)/otel_gpu_v1_usdt.h
	$(CXX) $(CXXFLAGS) -D__HIP_PLATFORM_AMD__ -c rocm_producer.cpp -I$(ROCM_PATH)/include \
	  -I$(CONTRACT) -I$(SAMPLER) -o rocm_producer.o
	$(CC) $(LDFLAGS) rocm_producer.o -L$(ROCM_PATH)/lib -lrocprofiler-sdk -lpthread -o $@
	rm -f rocm_producer.o

# The entry point must be exported, the library must not depend on libstdc++,
# and it must fire the probe from a single place, with the event in the pinned
# register and a semaphore.
check: $(OUT)
	@nm -D --defined-only $(OUT) | grep -qE ' T rocprofiler_configure$$' \
	  || { echo "$(OUT): rocprofiler_configure is not exported"; exit 1; }
	@! readelf -d $(OUT) | grep NEEDED | grep -q stdc++ \
	  || { echo "$(OUT): depends on libstdc++"; exit 1; }
	@$(MAKE) --no-print-directory -C $(SAMPLER) check-usdt SO=$(CURDIR)/$(OUT)

IMG ?= rocm/dev-ubuntu-24.04:7.0

docker:
	docker run --rm --platform linux/amd64 -v "$(CURDIR)/../..":/src \
	  -w /src/tools/rocm-producer $(IMG) bash -c '\
	  apt-get update -qq && apt-get install -y -qq gcc g++ make systemtap-sdt-dev binutils >/dev/null && \
	  make check'

clean:
	rm -f *.so *.o

.PHONY: check docker clean
