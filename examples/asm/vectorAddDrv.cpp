#include <stdio.h>
#include <string>
#include <iostream>
#include <cstring>
#include <fstream>
#include <streambuf>
#include <cuda.h>
#include <cmath>
#include <vector>

#define CHK(X) if ((err = X) != CUDA_SUCCESS) printf("CUDA error %d at %d\n", (int)err, __LINE__)

CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;
CUfunction vecAdd_kernel;
CUresult err;
CUdeviceptr d_A;
CUdeviceptr d_B;
CUdeviceptr d_C;

int main(int argc, char** argv) {
    printf("Vector Addition (Driver API)\n");
    int N = 50000, devID = 0;
    size_t  size = N * sizeof(float);

    // Initialize
    CHK(cuInit(0));
    CHK(cuDeviceGet(&cuDevice, devID));
    // Create context
    CHK(cuCtxCreate(&cuContext, 0, cuDevice));
    // Load PTX file
    std::ifstream my_file("vectorAdd_kernel.ptx");
    std::string my_ptx((std::istreambuf_iterator<char>(my_file)), std::istreambuf_iterator<char>());
    // Create module from PTX
    CHK(cuModuleLoadData(&cuModule, my_ptx.c_str()));

    // Get function handle from module
    CHK(cuModuleGetFunction(&vecAdd_kernel, cuModule, "VecAdd_kernel"));

    // Allocate/initialize vectors in host memory
    std::vector<float> h_A(N, 1.0f);
    std::vector<float> h_B(N, 2.0f);
    std::vector<float> h_C(N);

    // Allocate vectors in device memory
    CHK(cuMemAlloc(&d_A, size));
    CHK(cuMemAlloc(&d_B, size));
    CHK(cuMemAlloc(&d_C, size));

    // Copy vectors from host memory to device memory
    CHK(cuMemcpyHtoD(d_A, h_A.data(), size));
    CHK(cuMemcpyHtoD(d_B, h_B.data(), size));

    // Grid/Block configuration
    int threadsPerBlock = 256;
    int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;

    void *args[] = { &d_A, &d_B, &d_C, &N };

    // Launch the CUDA kernel
    CHK(cuLaunchKernel(vecAdd_kernel,  blocksPerGrid, 1, 1,
                               threadsPerBlock, 1, 1,
                               0,
                               NULL, args, NULL));

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    CHK(cuMemcpyDtoH(h_C.data(), d_C, size));

    // Verify result
    for (int i = 0; i < N; ++i)
    {
        float sum = h_A[i] + h_B[i];
        if (fabs(h_C[i] - sum) > 1e-7f)
        {
            printf("mismatch!");
            break;
        }
    }
    return 0;
}