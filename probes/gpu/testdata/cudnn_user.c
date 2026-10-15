// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

// Calls the fake cuDNN after waiting a second for the profiler to attach, then
// prints the number of calls.

#include <stdio.h>
#include <unistd.h>

enum { calls = 200 };

int cudnnConvolutionForward(void);

int main(void)
{
  sleep(1);
  for (int i = 0; i < calls; i++) {
    cudnnConvolutionForward();
  }
  printf("%d\n", calls);
  return 0;
}
