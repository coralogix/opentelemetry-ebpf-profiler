// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0
//
// kinds_app — exercises the kinds of events of the otel_gpu_v1 contract with
// known amounts, for end-to-end checks on a GPU host. On each of the first
// `devices` GPUs it runs kernels, alone and in a CUDA graph, memory copies in
// three directions and to and from a CUDA array, through the runtime and the
// driver API, memory fills, and a kernel whose name is longer than an event's
// payload. With OTEL_GPU_SAMPLE_RATE=0 the profile must show exactly the
// totals it prints as "expect <profile type> <leaf> <value>", and the kernel
// with the long name, whose leaf it prints as "expect long-name <prefix>".
//
// It also launches kernels from the GPU, which no host call starts, and
// prints their name as "expect device-launch <leaf>".
//
// Build: nvcc -O2 -rdc=true -o kinds_app kinds_app.cu -lcuda -lcudadevrt
// Usage: kinds_app <seconds to wait for the profiler> [devices]
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <utility>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

constexpr size_t kBytes         = 16 << 20; // per buffer
constexpr int    kRepeats       = 20;
constexpr int    kKernels       = 500;
constexpr int    kGraphKernels  = 8;
constexpr int    kGraphLaunches = 50;
// A CUDA array of kArrayWidth by kArrayHeight floats, for the runtime's
// copies, and one of kLineWidth floats, for the driver's, which take 1D arrays.
constexpr size_t kArrayWidth    = 1024;
constexpr size_t kArrayHeight   = 1024;
constexpr size_t kArrayBytes    = kArrayWidth * kArrayHeight * sizeof(float);
constexpr size_t kLineWidth     = 65536;
constexpr size_t kLineBytes     = kLineWidth * sizeof(float);
constexpr int    kLongNames     = 10;
constexpr int    kParents       = 10;
constexpr int    kChildren      = 20;

static void check(cudaError_t err, const char *what)
{
  if (err != cudaSuccess) {
    fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(err));
    exit(1);
  }
}

__global__ void spin(float *x, size_t n, int iterations)
{
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) {
    float v = x[i];
    for (int k = 0; k < iterations; k++) {
      v = v * 1.0001f + 0.001f;
    }
    x[i] = v;
  }
}

static void launch_spin(float *x, size_t n, cudaStream_t stream)
{
  spin<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(x, n, 2000);
}

static void check_driver(CUresult err, const char *what)
{
  if (err != CUDA_SUCCESS) {
    const char *msg = "unknown error";
    cuGetErrorString(err, &msg);
    fprintf(stderr, "%s: %s\n", what, msg);
    exit(1);
  }
}

// Instantiated with a type of a thousand template arguments, its mangled name
// is longer than an event's payload.
template <typename T> __global__ void long_name(float *x)
{
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    x[0] += 1.0f;
  }
}
using LongType = std::make_integer_sequence<int, 1000>;

__global__ void child(float *x)
{
  if (threadIdx.x == 0) {
    x[0] += 1.0f;
  }
}

// parent launches child kernels from the GPU.
__global__ void parent(float *x, int children)
{
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    for (int i = 0; i < children; i++) {
      child<<<1, 32>>>(x);
    }
  }
}

static void run_device(int device)
{
  const size_t n = kBytes / sizeof(float);
  check(cudaSetDevice(device), "cudaSetDevice");
  nvtxRangePushA("kinds_app");

  float *a, *b;
  void *host;
  check(cudaMalloc(&a, kBytes), "cudaMalloc");
  check(cudaMalloc(&b, kBytes), "cudaMalloc");
  check(cudaMallocHost(&host, kBytes), "cudaMallocHost");

  for (int r = 0; r < kRepeats; r++) {
    check(cudaMemset(a, 0, kBytes), "cudaMemset");
    check(cudaMemcpy(a, host, kBytes, cudaMemcpyHostToDevice), "HtoD");
    check(cudaMemcpy(host, a, kBytes, cudaMemcpyDeviceToHost), "DtoH");
    check(cudaMemcpy(b, a, kBytes, cudaMemcpyDeviceToDevice), "DtoD");
  }

  cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
  cudaArray_t array, line;
  check(cudaMallocArray(&array, &desc, kArrayWidth, kArrayHeight), "cudaMallocArray");
  check(cudaMallocArray(&line, &desc, kLineWidth), "cudaMallocArray");
  const size_t pitch = kArrayWidth * sizeof(float);
  for (int r = 0; r < kRepeats; r++) {
    check(cudaMemcpy2DToArray(array, 0, 0, host, pitch, pitch, kArrayHeight,
                              cudaMemcpyHostToDevice), "runtime HtoA");
    check(cudaMemcpy2DFromArray(host, pitch, array, 0, 0, pitch, kArrayHeight,
                                cudaMemcpyDeviceToHost), "runtime AtoH");
    check_driver(cuMemcpyHtoA((CUarray)line, 0, host, kLineBytes), "driver HtoA");
    check_driver(cuMemcpyAtoH(host, (CUarray)line, 0, kLineBytes), "driver AtoH");
  }
  check(cudaFreeArray(array), "cudaFreeArray");
  check(cudaFreeArray(line), "cudaFreeArray");

  for (int k = 0; k < kLongNames; k++) {
    long_name<LongType><<<1, 32>>>(a);
  }
  for (int k = 0; k < kParents; k++) {
    parent<<<1, 1>>>(a, kChildren);
  }

  for (int k = 0; k < kKernels; k++) {
    launch_spin(a, n, 0);
  }

  cudaStream_t stream;
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  check(cudaStreamCreate(&stream), "cudaStreamCreate");
  check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "capture");
  for (int k = 0; k < kGraphKernels; k++) {
    launch_spin(b, n, stream);
  }
  check(cudaStreamEndCapture(stream, &graph), "capture end");
  check(cudaGraphInstantiate(&exec, graph, 0), "cudaGraphInstantiate");
  for (int l = 0; l < kGraphLaunches; l++) {
    check(cudaGraphLaunch(exec, stream), "cudaGraphLaunch");
  }
  check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

  check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

  check(cudaFree(a), "cudaFree");
  check(cudaFree(b), "cudaFree");
  check(cudaFreeHost(host), "cudaFreeHost");
  nvtxRangePop();

  cudaDeviceProp prop;
  check(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
  const unsigned char *u = (const unsigned char *)prop.uuid.bytes;
  printf("gpu %d GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x\n",
         device, u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8], u[9], u[10], u[11],
         u[12], u[13], u[14], u[15]);
}

int main(int argc, char **argv)
{
  const int wait = argc > 1 ? atoi(argv[1]) : 20;
  int devices    = argc > 2 ? atoi(argv[2]) : 1;
  int available  = 0;
  check(cudaGetDeviceCount(&available), "cudaGetDeviceCount");
  if (devices > available) {
    devices = available;
  }

  check(cudaFree(0), "cudaFree"); // start CUDA, so the producer loads
  sleep(wait);

  for (int d = 0; d < devices; d++) {
    run_device(d);
  }

  const long long repeated = (long long)kRepeats * kBytes * devices;
  // Copies to and from arrays, through the runtime and through the driver,
  // count as copies to and from the GPU's memory.
  const long long arrays = (long long)kRepeats * (kArrayBytes + kLineBytes) * devices;
  printf("expect gpu_mem_bytes memcpy:HtoD %lld\n", repeated + arrays);
  printf("expect gpu_mem_bytes memcpy:DtoH %lld\n", repeated + arrays);
  printf("expect gpu_mem_bytes memcpy:DtoD %lld\n", repeated);
  printf("expect gpu_mem_bytes memset %lld\n", repeated);
  printf("expect kernels spin %d\n", (kKernels + kGraphKernels * kGraphLaunches) * devices);
  printf("expect long-name _Z9long_nameISt16integer_sequenceIiJLi0ELi1ELi2E\n");
  printf("expect device-launch child\n");
  return 0;
}
