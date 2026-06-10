#include <cstdio>
#include <nvtx3/nvToolsExt.h>
#include <cuda_runtime.h>
#include <chrono>
__global__ void busy(float* x,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float v=x[i];for(int k=0;k<4000;k++)v=v*1.0001f+0.001f;x[i]=v;}}
int main(){
  int n=1<<20; float* d; cudaMalloc(&d,n*sizeof(float)); cudaMemset(d,0,n*sizeof(float));
  busy<<<(n+255)/256,256>>>(d,n); cudaDeviceSynchronize();
  printf("nvtx app start\n"); fflush(stdout);
  auto t0=std::chrono::steady_clock::now();
  long it=0;
  while(std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count() < 40.0){
    nvtxRangePushA("vecop_range");
    busy<<<(n+255)/256,256>>>(d,n);
    nvtxRangePop();
    if((++it%500)==0) cudaDeviceSynchronize();
  }
  cudaDeviceSynchronize();
  printf("done it=%ld\n",it); return 0;
}
