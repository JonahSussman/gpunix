#include <iostream>

extern "C" {

#include "a_function.cuh"

__global__ void kernel(int* y) {
  a_function(y);
}

} // extern "C"

int main(void) {
  int* y;
  cudaMallocManaged(&y, 10 * sizeof(int));

  kernel<<<1, 1>>>(y);
  cudaDeviceSynchronize();
  for (int i = 0; i < 10; i++) {
    std::cout << y[i] << " ";
  }

  return 0;
}
