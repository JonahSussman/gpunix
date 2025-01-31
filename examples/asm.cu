#include <bits/stdc++.h>

__global__ void asm_examples() {
  // int i = 10;
  // int j = 20; 
  // int k = 30;
  
  // printf("before: i = %d, j = %d, k = %d\n", i, j, k);

  // // "h" = .u16 reg
  // // "r" = .u32 reg
  // // "l" = .u64 reg
  // // "f" = .f32 reg
  // // "d" = .f64 reg


  // // "r"  - register is read from
  // // "=r" - register is written to
  // // "+r" - register is both read from and written to

  // asm("add.s32 %0, %1, %2;" : "=r"(i) : "r"(j), "r"(k));
  // // ls.s32 r1, [j];
  // // ls.s32 r2, [k];
  // // add.s32 r3, r1, r2;
  // // st.s32 [i], r3;

  // printf("after:  i = %d, j = %d, k = %d\n", i, j, k);

  // // Best practice to put \n\t at the end of the asm statement
  // int y;
  // int x = 10;
  // asm(
  //   ".reg .u32 t1;\n\t"           // temp reg t1
  //   " mul.lo.u32 t1, %1, %1;\n\t" // t1 = x * x
  //   " mul.lo.u32 %0, t1, %1;\n\t" // y  = t1 * x
  //   : "=r"(y) : "r"(x)                     // output
  // );

  // printf("the cube of %d is %d\n", x, y);

  uint32_t sp_32;
  uint64_t sp_64;
  printf("address of sp_32: %p\n", &sp_32);
  printf("address of sp_64: %p\n", &sp_64);

  asm("stacksave.u32 %0;" : "=r"(sp_32));
  asm("stacksave.u64 %0;" : "=l"(sp_64));

  // printf("stack save 32: %x\n", sp_32);
  // printf("stack save 64: %lu\n", sp_64);
  

  // printf("%x", *(char*)(sp_32 & 0xffffff));

  printf("sp_64: %lx\n", sp_64);
  printf("&sp_64: %p\n", &sp_64);
  printf("*(&sp_64): %lx\n", *(&sp_64));
}

int main(int argc, char** argv) {
  asm_examples<<<1, 1>>>();
  cudaDeviceSynchronize();
}