// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0
//
// Checks of the call site sampler on simulated call sequences: the sampled
// rate, and that the weighted samples estimate the totals. The generator is
// seeded, so results are deterministic.

#include "sampler.h"

#include <math.h>
#include <stdio.h>

#define NS_PER_S 1000000000ull
#define RATE     200

static int failures;

static void expect(int ok, const char *what, double got, double want)
{
  if (!ok) {
    fprintf(stderr, "FAIL: %s: got %.1f, want %.1f\n", what, got, want);
    failures++;
  }
}

static void expect_near(const char *what, double got, double want, double tolerance)
{
  expect(fabs(got - want) <= tolerance * want, what, got, want);
}

// A training step: a kernel after a long host-side pause, then a burst of
// short launches. Sampling that favors call sites after a pause, like a token
// bucket, overestimates the first kernel several times.
enum { step_kernels = 7 };
static const uint64_t step_gaps_ns[step_kernels] = {
  1000000, 10000, 10000, 10000, 10000, 10000, 10000};

static void test_steps(void)
{
  struct otel_sampler s = {0};
  uint64_t rng          = 42;
  uint64_t now          = NS_PER_S;
  const uint64_t end    = now + 60 * NS_PER_S;

  double calls[step_kernels]    = {0};
  double estimate[step_kernels] = {0};
  uint64_t sampled_in_second    = 0;
  uint64_t second               = now / NS_PER_S;
  while (now < end) {
    for (int k = 0; k < step_kernels; k++) {
      now += step_gaps_ns[k];
      if (now / NS_PER_S != second) {
        // The first second ramps up from no history.
        if (second > 1) {
          expect(
            sampled_in_second >= RATE / 2 && sampled_in_second <= RATE * 3 / 2,
            "sampled call sites per second",
            (double)sampled_in_second,
            RATE);
        } else {
          expect(
            sampled_in_second <= 5 * RATE,
            "sampled call sites in the first second",
            (double)sampled_in_second,
            5 * RATE);
        }
        second            = now / NS_PER_S;
        sampled_in_second = 0;
      }
      calls[k]++;
      const uint32_t weight = otel_sampler_weight(&s, RATE, now, &rng);
      if (weight != 0) {
        estimate[k] += weight;
        sampled_in_second++;
      }
    }
  }
  for (int k = 0; k < step_kernels; k++) {
    expect_near("estimated call sites of a kernel", estimate[k], calls[k], 0.1);
  }
}

// Call sites below the rate are all sampled, each standing for itself.
static void test_below_rate(void)
{
  struct otel_sampler s = {0};
  uint64_t rng          = 7;
  int all_once          = 1;
  for (uint64_t now = NS_PER_S; now < 11 * NS_PER_S; now += NS_PER_S / (RATE / 2)) {
    all_once = all_once && otel_sampler_weight(&s, RATE, now, &rng) == 1;
  }
  expect(all_once, "every call site sampled once below the rate", 0, 1);
}

// A rate of 0 samples every call site.
static void test_disabled(void)
{
  struct otel_sampler s = {0};
  uint64_t rng          = 7;
  int all_once          = 1;
  for (uint64_t now = NS_PER_S; now < 2 * NS_PER_S; now += 10000) {
    all_once = all_once && otel_sampler_weight(&s, 0, now, &rng) == 1;
  }
  expect(all_once, "every call site sampled once with sampling disabled", 0, 1);
}

// Bursts separated by pauses: the estimate holds across the rate changes.
static void test_bursts(void)
{
  struct otel_sampler s = {0};
  uint64_t rng          = 3;
  uint64_t now          = NS_PER_S;
  double calls = 0, estimate = 0;
  for (int burst = 0; burst < 20; burst++) {
    const uint64_t burst_end = now + 3 * NS_PER_S;
    for (; now < burst_end; now += 50000) {
      calls++;
      estimate += otel_sampler_weight(&s, RATE, now, &rng);
    }
    now += 5 * NS_PER_S;
  }
  expect_near("estimated call sites across bursts", estimate, calls, 0.05);
}

int main(void)
{
  test_steps();
  test_below_rate();
  test_disabled();
  test_bursts();
  if (failures != 0) {
    return 1;
  }
  printf("sampler: ok\n");
  return 0;
}
