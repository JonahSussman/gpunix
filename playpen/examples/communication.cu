#include <iostream>
#include <cstdio>
#include <cuda_runtime.h>

using namespace std;

__global__ void test(
  volatile int *flag, 
  volatile int *data_ready, 
  volatile int *data
) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  while (true) {
    if (*flag == 0) {
      // wait for data transfer
      while (true) {
        if (*data_ready == 0) {
          printf("x");
        }
        else {
          break;
        }
      }
      printf("data %d\n", *data);
      __syncthreads();
    }
    else {
      break;
    }
  }

  printf("gpu finish %d\n", tid);
}

int main() {
  int attr = 0;
  cudaDeviceGetAttribute(&attr, cudaDevAttrConcurrentManagedAccess, 0);
  if (attr == 0) {
    printf("GPU does not support cudaDevAttrConcurrentManagedAccess\n"); 
    return 0;
  }

  // flags
  int *flag;
  cudaMallocManaged(&flag, sizeof(int));
  *flag = 0;

  int *data_ready;
  cudaMallocManaged(&data_ready, sizeof(int));
  *data_ready = 0;

  // data
  int *data = (int*)malloc(sizeof(int));
  int *data_device;
  *data = 777;
  cudaMalloc(&data_device, sizeof(int));
  cudaMemcpy(data_device, data, sizeof(int), cudaMemcpyHostToDevice);
  cudaStream_t s1, s2;
  cudaStreamCreate(&s1);
  cudaStreamCreate(&s2);

  // launch kernel
  int block = 8, grid = 1;
  test<<<grid, block, 0, s1>>> (flag, data_ready, data_device);

  // random host code
  for (int i = 0; i < 1e5; i++);
  printf("host do something\n");

  // update data
  *data = 987;
  cudaMemcpyAsync(data_device, data, sizeof(int), cudaMemcpyHostToDevice, s2);
  printf("host copied\n");
  *data_ready = 1;

  // update flag
  *flag = 1;

  cudaDeviceSynchronize();

  // free memory
  cudaFree(flag);

  printf("host finish\n");
}