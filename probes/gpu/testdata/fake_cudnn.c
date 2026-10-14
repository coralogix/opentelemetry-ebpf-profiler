// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

// Stands in for cuDNN in the integration tests: exports one instrumented
// function, which takes at least 500 µs.

#include <unistd.h>

int cudnnConvolutionForward(void)
{
  usleep(500);
  return 0;
}
