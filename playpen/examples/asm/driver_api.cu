// Build with:
// nvcc driver_api.cu -arch=sm_61 -o driver_api.out -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcuda

// Driver API
// https://stackoverflow.com/questions/22639097/what-is-the-difference-between-the-cuda-api-cu-and-cuda


#include <cuda_runtime_api.h>
#include <cuda.h>

#include <iostream>
#include <fstream>
#include <vector>


CUresult err;

#define CHK(X) if ((err = X) != CUDA_SUCCESS) printf("CUDA error %d at %d\n", (int)err, __LINE__)

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


  CUcontext context;
  CUdevice cuDevice;

  // These next few lines simply initialize your work with the CUDA driver,
  // they're not specific to PTX compilation
  CHK(cuInit(0));
  CHK(cuDeviceGet(&cuDevice, 0)); // or some other device on your system
  CHK(cuCtxCreate(&context, 0, cuDevice));

  // The magic happens here:
  CUmodule module;
  cuModuleLoadDataEx(&module, ptx_string.c_str(), 0, 0, 0);

  // And here is how you use your compiled PTX
  CUfunction VecAdd_kernel;
  cuModuleGetFunction(&VecAdd_kernel, module, "VecAdd_kernel");

  // Allocate/initialize vectors in host memory
  int N = 1024;
  size_t size = N * sizeof(float);

  std::vector<float> h_A(N, 1.0f);
  std::vector<float> h_B(N, 2.0f);
  std::vector<float> h_C(N);

  // Allocate vectors in device memory
  CUdeviceptr d_A, d_B, d_C;
  CHK(cuMemAlloc(&d_A, size));
  CHK(cuMemAlloc(&d_B, size));
  CHK(cuMemAlloc(&d_C, size));

  // Copy vectors from host memory to device memory
  CHK(cuMemcpyHtoD(d_A, h_A.data(), size));
  CHK(cuMemcpyHtoD(d_B, h_B.data(), size));

  const int threadsPerBlock = 256;
  const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  void* args[] = { &d_A, &d_B, &d_C, &N };

  CHK(cuLaunchKernel(
    VecAdd_kernel,
    blocksPerGrid, 1, 1,
    threadsPerBlock, 1, 1,
    0,
    NULL,
    args, NULL
  ));

  // Copy result from device memory to host memory
  CHK(cuMemcpyDtoH(h_C.data(), d_C, size));

  // Verify result
  for (int i = 0; i < N; ++i) {
    if (h_C[i] != h_A[i] + h_B[i]) {
      std::cerr << "Error: C[" << i << "] = " << h_C[i] << std::endl;
      break;
    }
  }

  std::cout << "Result verified!" << std::endl;
}