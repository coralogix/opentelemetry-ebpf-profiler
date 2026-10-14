// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0
//
// Call site sampling. Each thread samples each class of call sites on its own,
// aiming at `rate` sampled call sites per second. A call site is sampled with
// probability 1/N and then stands for N call sites, where N grows with the call
// sites seen in the last second. N depends only on earlier call sites, so the
// weighted samples estimate the totals without bias, also while the call rate
// changes, and unlike a token bucket the choice does not favor call sites that
// follow a pause.
#pragma once

#include <stdint.h>

#define OTEL_SAMPLER_WINDOW_NS 1000000000ull

struct otel_sampler {
  uint64_t window_start; // ns
  uint64_t current;      // call sites in the current window
  uint64_t previous;     // call sites in the previous window
};

// otel_rng_next steps a xorshift64* generator, whose state must not be 0.
static inline uint64_t otel_rng_next(uint64_t *state)
{
  uint64_t x = *state;
  x ^= x >> 12;
  x ^= x << 25;
  x ^= x >> 27;
  *state = x;
  return x * 0x2545f4914f6cdd1dull;
}

// otel_sampler_weight returns the sample weight of the call site at now_ns, or
// 0 if it is not sampled. A rate of 0 samples every call site.
static inline uint32_t
otel_sampler_weight(struct otel_sampler *s, uint32_t rate, uint64_t now_ns, uint64_t *rng)
{
  if (rate == 0) {
    return 1;
  }
  uint64_t elapsed = now_ns - s->window_start;
  if (elapsed >= OTEL_SAMPLER_WINDOW_NS) {
    const int adjacent = elapsed < 2 * OTEL_SAMPLER_WINDOW_NS;
    s->previous        = adjacent ? s->current : 0;
    s->current         = 0;
    s->window_start    = adjacent ? s->window_start + OTEL_SAMPLER_WINDOW_NS : now_ns;
    elapsed            = now_ns - s->window_start;
  }
  // Call sites of the last second: the previous window counts for the part
  // of it the last second still covers.
  const uint64_t recent =
    s->current + s->previous * (OTEL_SAMPLER_WINDOW_NS - elapsed) / OTEL_SAMPLER_WINDOW_NS;
  s->current++;

  const uint64_t period = recent / rate + 1;
  if (otel_rng_next(rng) % period != 0) {
    return 0;
  }
  return period < UINT32_MAX ? (uint32_t)period : UINT32_MAX;
}
