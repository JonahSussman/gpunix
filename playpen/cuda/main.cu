#include <bits/stdc++.h>

// The default on my machine for GTX 1060 is 1024 bytes
#define CUDA_LIMIT_STACK_SIZE 4096

// Default disk size is 1MB

__device__ void waitdisk(void) {
  while ((inb(0x1f7) & 0xc0) != 0x40);
}

int main(void) {
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, 0);

  cudaError_t err;

  // https://forums.developer.nvidia.com/t/what-is-the-maximum-cuda-stack-frame-size-per-kerenl/31449/2
  // stack frame size available per thread =
  //    min (amount of local memory per thread as documented in section G.1 table 12,
  //    available GPU memory / number of SMs / maximum resident threads per SM)

  err = cudaDeviceSetLimit(cudaLimitStackSize, CUDA_LIMIT_STACK_SIZE);
  if (err != cudaSuccess) {
    printf("Error setting stack size: %s\n", cudaGetErrorString(err));
    return -1;
  }

  size_t* stack_size = new size_t;

  err = cudaDeviceGetLimit(stack_size, cudaLimitStackSize);
  if (err != cudaSuccess) {
    printf("Error getting stack size: %s\n", cudaGetErrorString(err));
    return -1;
  }

  printf("CUDA stack size limit set to %zu bytes\n", *stack_size);

  // Boot (bootasm.s and bootmain.c) 
  
  // In a normal operating system, we load the kernel from the first sector of
  // the disk (usually 512 bytes). We have the advantage of being able to load
  // everything via this cuda program.

  


  return 0;
}