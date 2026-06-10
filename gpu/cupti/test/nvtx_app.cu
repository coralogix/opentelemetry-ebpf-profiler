// SPDX-License-Identifier: Apache-2.0
//
// Test workload: launches a busy kernel in a loop for ~40s, each launch
// wrapped in an NVTX range, to exercise the shim's NVTX correlation path
// (kernel samples should carry an nvtx:vecop_range frame).

#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

__global__ void busy(float *x, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = x[i];
    for (int k = 0; k < 4000; k++) {
      v = v * 1.0001f + 0.001f;
    }
    x[i] = v;
  }
}

int main()
{
  int n = 1 << 20;
  float *d;
  cudaMalloc(&d, n * sizeof(float));
  cudaMemset(d, 0, n * sizeof(float));

  // Warm up: first launch pays module load / context setup.
  busy<<<(n + 255) / 256, 256>>>(d, n);
  cudaDeviceSynchronize();

  printf("nvtx app start\n");
  fflush(stdout);

  auto t0 = std::chrono::steady_clock::now();
  long it = 0;
  while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 40.0) {
    nvtxRangePushA("vecop_range");
    busy<<<(n + 255) / 256, 256>>>(d, n);
    nvtxRangePop();
    // Sync occasionally so the launch queue doesn't grow unboundedly.
    if ((++it % 500) == 0) {
      cudaDeviceSynchronize();
    }
  }
  cudaDeviceSynchronize();

  printf("done it=%ld\n", it);
  return 0;
}
