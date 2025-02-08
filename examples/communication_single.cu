#include <bits/stdc++.h>
#include <cuda_runtime.h>

// Kind of a mix between c and c++ RAII I know, but __managed__ variables can't
// have dynamic constructors.
struct Channel {
  volatile int* read_ready;
  volatile int* write_ready;
  volatile char* data;
  volatile int* size;

  __host__ __device__ void write(void* src, int sz) {
    // TODO: Make atomic
    while (*write_ready == 0);

    // copy data
    for (int i = 0; i < sz; i++) {
      ((char*)data)[i] = ((char*)src)[i];
    }

    *size = sz;
    *write_ready = 0;
    *read_ready = 1;

    #ifndef __CUDA_ARCH__
    cudaDeviceSynchronize();
    #endif
  }

  __host__ __device__ void read(void* dst, int sz) {
    // TODO: Make atomic
    while (*read_ready == 0);

    // copy data
    for (int i = 0; i < sz; i++) {
      ((char*)dst)[i] = ((char*)data)[i];
    }

    #ifdef __CUDA_ARCH__
    __syncthreads();
    #endif

    *read_ready = 0;
    *write_ready = 1;

    #ifndef __CUDA_ARCH__
    cudaDeviceSynchronize();
    #endif
  }
};


void channel_init(Channel* ch) {
  cudaMallocManaged(&ch->read_ready, sizeof(int));
  cudaMallocManaged(&ch->write_ready, sizeof(int));
  cudaMallocManaged(&ch->data, sizeof(char)*1024*1024); // 1MB
  cudaMallocManaged(&ch->size, sizeof(int));

  *ch->read_ready = 0;
  *ch->write_ready = 1;
  *ch->size = 0;
}

void channel_free(Channel* ch) {
  cudaFree((void*)ch->read_ready);
  cudaFree((void*)ch->write_ready);
  cudaFree((void*)ch->data);
  cudaFree((void*)ch->size);
}


__managed__ Channel ch;


struct ComplexThing {
  int a;
  char b;
  float c;
};

__global__ void channel_test() {
  int thread_id = blockDim.x * blockIdx.x + threadIdx.x;

  ComplexThing recieved;
  ch.read(&recieved, sizeof(ComplexThing));

  printf("[device] data received:\n"
         "[device]   a: %d\n"
         "[device]   b: %c\n"
         "[device]   c: %f\n", recieved.a, recieved.b, recieved.c);

  recieved.a += 1;
  recieved.b += 1;
  recieved.c += 1;
  printf("[device] data modified:\n"
         "[device]   a: %d\n"
         "[device]   b: %c\n"
         "[device]   c: %f\n", recieved.a, recieved.b, recieved.c);
  ch.write(&recieved, sizeof(ComplexThing));
  
  printf("[device] finished\n");
}

int main(void) {
  int attr = 0;
  cudaDeviceGetAttribute(&attr, cudaDevAttrConcurrentManagedAccess, 0);
  if (attr == 0) {
    printf("GPU does not support cudaDevAttrConcurrentManagedAccess\n");
    return 0;
  }

  // Channel ch;
  channel_init(&ch);
  
  cudaStream_t kernel_stream, data_stream;
  cudaStreamCreate(&kernel_stream);
  cudaStreamCreate(&data_stream);

  const int threads_per_block = 1;
  const int blocks_per_grid = 1;

  channel_test<<<blocks_per_grid, threads_per_block, 0, kernel_stream>>>();

  ComplexThing data;
  data.a = 123;
  data.b = 'x';
  data.c = 3.14;

  printf("[host] data to send:\n"
         "[host]   a: %d\n"
         "[host]   b: %c\n"
         "[host]   c: %f\n", data.a, data.b, data.c);

  ch.write(&data, sizeof(ComplexThing));
  ch.read(&data, sizeof(ComplexThing));

  printf("[host] data received:\n"
         "[host]   a: %d\n"
         "[host]   b: %c\n"
         "[host]   c: %f\n", data.a, data.b, data.c);

  printf("[host] finished\n");

  cudaStreamSynchronize(kernel_stream);
  cudaStreamSynchronize(data_stream);
  cudaStreamDestroy(kernel_stream);
  cudaStreamDestroy(data_stream);

  channel_free(&ch);
  return 0;
}