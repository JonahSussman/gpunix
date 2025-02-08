#include <string>
#include <iostream>
#include <stdexcept>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

cudaError_t cuda_err;
nvrtcResult nvrtc_err;

#define CHK_CUDA(X) \
  if ((cuda_err = X) != cudaSuccess) \
    printf("CUDA error %d at %d\n", (int)cuda_err, __LINE__)

#define CHK_NVRTC(X) \
  if ((nvrtc_err = X) != NVRTC_SUCCESS) \
    printf("NVRTC error %d at %d\n", (int)nvrtc_err, __LINE__)

std::string compile_to_ptx(const std::string& source_code) {
    // Compile the source code to PTX using nvrtc
    nvrtcProgram prog;
    CHK_NVRTC(nvrtcCreateProgram(
      &prog, 
      source_code.c_str(), 
      "kernel.cu", 
      0, 
      nullptr, 
      nullptr
    ));
    
    nvrtcResult compile_result = nvrtcCompileProgram(prog, 0, nullptr);
    if (compile_result != NVRTC_SUCCESS) {
        size_t log_size;
        nvrtcGetProgramLogSize(prog, &log_size);
        std::string log(log_size, '\0');
        nvrtcGetProgramLog(prog, &log[0]);
        throw std::runtime_error("Compilation failed: " + log);
    }
    
    size_t ptx_size;
    nvrtcGetPTXSize(prog, &ptx_size);
    std::string ptx(ptx_size, '\0');
    nvrtcGetPTX(prog, &ptx[0]);
    
    nvrtcDestroyProgram(&prog);
    
    return ptx;
}