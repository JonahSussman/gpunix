/*
Demonstration of an async-await style cooperative multitasking that's required
because CUDA doesn't support pre-empting threads. However, it appears that
cuda-gdb does support pre-empting threads, but I haven't been able to dig into
the source code to see how it works. I suspect that they're doing some clever
stuff because they own the drivers.
*/

#include <bits/stdc++.h>
#include <nvfunctional>

struct RuntimeState {
  nvstd::function<void(RuntimeState*)> call_stack[64];
  int call_stack_size = 0;
  char stack[1024];

  // Requires at least CUDA 8
  // https://developer.nvidia.com/blog/new-compiler-features-cuda-8/
  nvstd::function<void(int)> f; 
};


__device__ void the_function(RuntimeState* state);
__device__ void the_other_function(RuntimeState* state);


__device__ void the_function(RuntimeState* state) {
  printf("I am in the_function!\n");

  int q = 1000;

  auto after = [=] __device__ (RuntimeState* state) {
    printf("I am in the_after_function!\n");
    printf("%d\n", q);

    // Clear this function from the call stack and return to the previous function
    state->call_stack[state->call_stack_size - 1] = nullptr;
    state->call_stack_size--;
  };

  printf("after size: %d\n", sizeof(after));

  state->call_stack[state->call_stack_size - 1] = after;
  state->call_stack_size++;
  state->call_stack[state->call_stack_size - 1] = the_other_function;
}


__device__ void the_other_function(RuntimeState* state) {
  printf("I am in the_other_function!\n");

  // Clear this function from the call stack and return to the previous function
  state->call_stack[state->call_stack_size - 1] = nullptr;
  state->call_stack_size--;
}



__global__ void async_runtime(RuntimeState* state) {
  // Check if the call stack is empty
  if (state->call_stack_size == 0) {
    // Initialize the call stack with the first function
    state->call_stack[state->call_stack_size++] = the_function;
  }

  auto& next_function = state->call_stack[state->call_stack_size - 1];
  next_function(state);
}


int main(void) {
  // Host copies
  RuntimeState* h_state = new RuntimeState;
  
  // Device copies
  RuntimeState* d_state;
  cudaMalloc((void**)&d_state, sizeof(RuntimeState));

  for (int i = 0; i < 10; i++) {
    cudaMemcpy(d_state, h_state, sizeof(RuntimeState), cudaMemcpyHostToDevice);
    async_runtime<<<1, 1>>>(d_state);

    cudaDeviceSynchronize();
    cudaMemcpy(h_state, d_state, sizeof(RuntimeState), cudaMemcpyDeviceToHost);

    sleep(1);
  }

  return 0;
}