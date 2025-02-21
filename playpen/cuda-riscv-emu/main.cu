#include <stdio.h>
#include <getopt.h>

#include <iostream>

#include "cuda_helpers.cuh"
#include "machine.cuh"

// The default on my machine for GTX 1060 is 1024 bytes
#define CUDA_LIMIT_STACK_SIZE 4096

#ifndef CONFIG_VERSION
  #define CONFIG_VERSION "unknown"
#endif


// TODO(JonahSussman): Comment what this does
enum BlockDeviceModeEnum {
  BF_MODE_RO,
  BF_MODE_RW,
  BF_MODE_SNAPSHOT,
};


__global__ void main_kernel() {

}

static struct option options[] = {
  { "help", no_argument, NULL, 'h' },
  { "ctrlc", no_argument },
  { "rw", no_argument },
  { "ro", no_argument },
  { "append", required_argument },
  { "build-preload", required_argument },
  { NULL },
};

void help(void) {
  printf("riscv-emu-cuda version " CONFIG_VERSION ".\n"
          "Adapted for CUDA by Jonah Sussman. Originally written by Fabrice Bellard.\n"
          "usage: riscv-emu-cuda [options] config_file\n"
          "options are:\n"
          "-m ram_size       set the RAM size in MB\n"
          "-rw               allow write access to the disk image (default=snapshot)\n"
          "-ctrlc            the C-c key stops the emulator instead of being sent to the\n"
          "                  emulated software\n"
          "-append cmdline   append cmdline to the kernel command line\n"
          "\n"
          "Console keys:\n"
          "Press C-a x to exit the emulator, C-a h to get some help.\n");
  exit(1);
}


int main(int argc, char** argv) {
  // Set up CUDA
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


  // TODO(JonahSussman): Parse command line arguments

  


  VirtMachineParams* vm_params = cudaNewManaged<VirtMachineParams>();

  main_kernel<<<1, 1>>>();
  cudaDeviceSynchronize();
}