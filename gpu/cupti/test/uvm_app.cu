// SPDX-License-Identifier: Apache-2.0
//
// uvm_app — unified-memory workload to exercise the shim's UVM counters
// (migrations + page faults). Allocates managed memory and ping-pongs it
// between CPU and GPU for ~20s.
//
// Build (static cudart, runs anywhere with a driver):
//   nvcc -O2 -cudart=static -o uvm_app uvm_app.cu
#include <cstdio>
#include <ctime>

__global__ void touch(float *a, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    a[i] = a[i] * 1.0001f + 1.0f;
  }
}

static double now_s() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec / 1e9;
}

int main() {
  const size_t n = 64 << 20; // 256 MB of floats
  float *a = nullptr;
  if (cudaMallocManaged(&a, n * sizeof(float)) != cudaSuccess) {
    fprintf(stderr, "cudaMallocManaged failed\n");
    return 1;
  }
  for (size_t i = 0; i < n; i++) {
    a[i] = 1.0f;
  }
  printf("uvm_app: ping-pong start\n");
  fflush(stdout);
  double end = now_s() + 20;
  float sum = 0;
  while (now_s() < end) {
    touch<<<(n + 255) / 256, 256>>>(a, n); // GPU faults + HtoD migration
    cudaDeviceSynchronize();
    for (size_t i = 0; i < n; i += 1024) { // CPU faults + DtoH migration
      sum += a[i];
      a[i] += 1.0f;
    }
  }
  printf("uvm_app: done (%f)\n", sum);
  cudaFree(a);
  return 0;
}
