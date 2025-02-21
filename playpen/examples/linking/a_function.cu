extern "C" {

#include "a_function.cuh"

__device__ void a_function(int* y) {
  for (int i = 0; i < 10; i++) {
    y[i] = 1;
    b_function(y);
  }
}

__device__ void b_function(int* y) {
  for (int i = 0; i < 10; i++) {
    y[i] = 2;
  }
}

} // extern "C"