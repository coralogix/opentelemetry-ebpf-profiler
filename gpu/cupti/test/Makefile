# SPDX-License-Identifier: Apache-2.0
# S1 USDT bridge-proof producer. Needs sys/sdt.h (systemtap-sdt-dev /
# systemtap-sdt-devel). No CUDA/GPU required.
CC ?= cc
CFLAGS ?= -O2 -g -Wall

usdt_producer: usdt_producer.c ../usdt_probes.h
	$(CC) $(CFLAGS) $< -o $@

# Verify the USDT notes were emitted into the ELF (fail, don't just print —
# a producer without notes makes the bridge proof pass vacuously):
check: usdt_producer
	@readelf -n usdt_producer | grep -qi stapsdt && echo "stapsdt notes ✓" \
	  || { echo "no stapsdt notes (missing sys/sdt.h?)" >&2; exit 1; }

clean:
	rm -f usdt_producer

.PHONY: check clean
