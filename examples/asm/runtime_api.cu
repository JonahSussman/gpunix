// Build with:
// nvcc runtime_api.cu -arch=sm_61 -o runtime_api.out -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcuda

// Runtime API
// Requires CUDA Runtime 12.8 or later
// https://developer.nvidia.com/blog/dynamic-loading-in-the-cuda-runtime/


#include <cuda_runtime_api.h>
#include <cuda.h>

#include <iostream>
#include <fstream>
#include <vector>

cudaError_t err;

#define CHK(X) if ((err = X) != cudaSuccess) printf("CUDA error %d at %d\n", (int)err, __LINE__)

int main(int argc, char *argv[]) {
  std::ifstream ptx_file("vectorAdd_kernel.ptx");
  if (!ptx_file) {
    std::cerr << "Error opening PTX file" << std::endl;
    return -1;
  }

  std::string ptx_string(
    (std::istreambuf_iterator<char>(ptx_file)), // Why surround with parens?
    std::istreambuf_iterator<char>()
  );

  std::cout << "PTX code:" << std::endl;
  std::cout << ptx_string << std::endl;

  cudaLibrary_t library;
  cudaKernel_t kernel;

  cudaLibraryLoadData(&library, ptx_string.c_str(), 0, 0, 0, 0, 0, 0);
  cudaLibraryGetKernel(&kernel, library, "VecAdd_kernel");

  // Allocate/initialize vectors in host memory
  int N = 1024;
  size_t size = N * sizeof(float);

  std::vector<float> h_A(N, 1.0f);
  std::vector<float> h_B(N, 2.0f);
  std::vector<float> h_C(N);

  float *d_A, *d_B, *d_C;
  CHK(cudaMalloc((void**)&d_A, size));
  CHK(cudaMalloc((void**)&d_B, size));
  CHK(cudaMalloc((void**)&d_C, size));
  CHK(cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice));
  CHK(cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice));

  const int threadsPerBlock = 256;
  const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  void* args[] = { &d_A, &d_B, &d_C, &N };

  CHK(cudaLaunchKernel(
    (void*)kernel,
    blocksPerGrid,   // grid dimensions
    threadsPerBlock, // block dimensions
    args,            // kernel arguments
    0,               // shared mem
    NULL             // stream
  ));

  // Copy result from device memory to host memory
  CHK(cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost));

  // Verify result
  for (int i = 0; i < N; ++i) {
    if (h_C[i] != h_A[i] + h_B[i]) {
      std::cerr << "Error: C[" << i << "] = " << h_C[i] << std::endl;
      break;
    }
  }

  std::cout << "Result verified!" << std::endl;
}