#include <stdio.h>
// #include <cuda_runtime.h>

extern "C" { // Prevents name mangling
  __device__ void cranberry(int a, int b);

  __device__ void hello(int* y) {
    int x = 0;
    x--;

    *y = x;

    cranberry(1, 2);

    // printf("hello world\n");
  }


  __global__ void kernel(int* y) {
    hello(y);
  }
}

// int main() {
//   int *y;
//   cudaMalloc((void**)&y, sizeof(int));

//   hello<<<1, 1>>>(y);
//   cudaDeviceSynchronize();
//   return 0;
// }