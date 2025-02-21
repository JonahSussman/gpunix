# nvcc -Xcompiler -fPIC -I. -c a_function.cu
# ld -o a_function.ro -r a_function.o -L/usr/local/cuda/lib64 -lcudart_static -lculibos
# ar rs liba_function.a a_function.ro

# nvcc -o main main.cu -I ./ -L. -la_function -lcudart_static -lculibos

# https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#nvcc-options-for-separate-compilation

# -clean https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#clean-targets-clean
mkdir -p keep

# nvcc -arch=sm_61 --device-c -keep -keep-dir=keep -v a_function.cu
# nvcc -arch=sm_61 --device-c -keep -keep-dir=keep main.cu

# nvcc -arch=sm_61 -o main.out -keep -keep-dir=keep main.o a_function.o



# nvcc -arch=sm_61 --device-c -keep -keep-dir=keep -v a_function.cu

# _NVVM_BRANCH_=nvvm
# _SPACE_= 
# _CUDART_=cudart
# _HERE_=/usr/local/cuda-12.8/bin
# _THERE_=/usr/local/cuda-12.8/bin
# _TARGET_SIZE_=
# _TARGET_DIR_=
# _TARGET_DIR_=targets/x86_64-linux
# TOP=/usr/local/cuda-12.8/bin/..
CICC_PATH=/usr/local/cuda-12.8/bin/../nvvm/bin
# NVVMIR_LIBRARY_DIR=/usr/local/cuda-12.8/bin/../nvvm/libdevice
# LD_LIBRARY_PATH=/usr/local/cuda-12.8/bin/../lib:/usr/local/cuda-12.8/lib64:/usr/local/cuda-12.8/lib64:/usr/local/cuda-12.8/lib64
# PATH=/usr/local/cuda-12.8/bin/../nvvm/bin:/usr/local/cuda-12.8/bin:/usr/local/cuda-12.8/bin:/usr/local/cuda-12.8/bin:/usr/local/cuda-12.8/bin:/home/jonah/.nvm/versions/node/v18.2.0/bin:/home/jonah/.local/bin:/home/jonah/bin:/usr/lib64/ccache:/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/sbin:/home/jonah/go/bin:/opt/riscv/bin:/home/jonah/go/bin:/opt/riscv/bin:/home/jonah/.config/Code/User/globalStorage/github.copilot-chat/debugCommand:/home/jonah/go/bin:/opt/riscv/bin
# INCLUDES="-I/usr/local/cuda-12.8/bin/../targets/x86_64-linux/include"  
# LIBRARIES="-L/usr/local/cuda-12.8/bin/../targets/x86_64-linux/lib/stubs -L/usr/local/cuda-12.8/bin/../targets/x86_64-linux/lib"
# CUDAFE_FLAGS=
# PTXAS_FLAGS=

# create .cpp4.ii file
gcc -D__CUDA_ARCH_LIST__=610 -D__NV_LEGACY_LAUNCH -E -x c++ -D__CUDACC__ \
  -D__NVCC__ -D__CUDACC_RDC__ \
  "-I/usr/local/cuda-12.8/bin/../targets/x86_64-linux/include" \
  -D__CUDACC_VER_MAJOR__=12 -D__CUDACC_VER_MINOR__=8 -D__CUDACC_VER_BUILD__=93 \
  -D__CUDA_API_VER_MAJOR__=12 -D__CUDA_API_VER_MINOR__=8 \
  -D__NVCC_DIAG_PRAGMA_SUPPORT__=1 -D__CUDACC_DEVICE_ATOMIC_BUILTINS__=1 \
  -include "cuda_runtime.h" -m64 "a_function.cu" -o "keep/a_function.cpp4.ii" 

# create .cudafe1.cpp and .module_id file
cudafe++ --c++17 --gnu_version=140201 --display_error_number \
  --orig_src_file_name "a_function.cu" --orig_src_path_name \
  "/home/jonah/projects/cuda-unix/examples/linking/a_function.cu" \
  --allow_managed  --device-c  --m64 --parse_templates --gen_c_file_name \
  "keep/a_function.cudafe1.cpp" --stub_file_name "a_function.cudafe1.stub.c" \
  --gen_module_id_file --module_id_file_name "keep/a_function.module_id" \
  "keep/a_function.cpp4.ii" 

# create .cpp1.ii file
gcc -D__CUDA_ARCH__=610 -D__CUDA_ARCH_LIST__=610 -D__NV_LEGACY_LAUNCH -E -x c++ \
  -DCUDA_DOUBLE_MATH_FUNCTIONS -D__CUDACC__ -D__NVCC__ -D__CUDACC_RDC__ \
  "-I/usr/local/cuda-12.8/bin/../targets/x86_64-linux/include" \
  -D__CUDACC_VER_MAJOR__=12 -D__CUDACC_VER_MINOR__=8 -D__CUDACC_VER_BUILD__=93 \
  -D__CUDA_API_VER_MAJOR__=12 -D__CUDA_API_VER_MINOR__=8 \
  -D__NVCC_DIAG_PRAGMA_SUPPORT__=1 -D__CUDACC_DEVICE_ATOMIC_BUILTINS__=1 \
  -include "cuda_runtime.h" -m64 "a_function.cu" -o "keep/a_function.cpp1.ii" 

# create .cudafe1.c .cudafe1.gpu .cudafe1.stub.c and .ptx file
"$CICC_PATH/cicc" --c++17 --gnu_version=140201 --display_error_number \
  --orig_src_file_name "a_function.cu" --orig_src_path_name \
  "/home/jonah/projects/cuda-unix/examples/linking/a_function.cu" \
  --allow_managed --device-c   -arch compute_61 -m64 --no-version-ident -ftz=0 \
  -prec_div=1 -prec_sqrt=1 -fmad=1 --include_file_name "a_function.fatbin.c" \
  -tused --module_id_file_name "keep/a_function.module_id" --gen_c_file_name \
  "keep/a_function.cudafe1.c" --stub_file_name "keep/a_function.cudafe1.stub.c"\
  --gen_device_file_name "keep/a_function.cudafe1.gpu" \
  "keep/a_function.cpp1.ii" -o "keep/a_function.ptx"

# create .sm_61.cubin file
ptxas -arch=sm_61 -m64 --compile-only  "keep/a_function.ptx" -o \
  "keep/a_function.sm_61.cubin" 

# create .fatbin.c and .fatbin file
fatbinary --create="keep/a_function.fatbin" -64 --cmdline="--compile-only  " \
  --cicc-cmdline="-ftz=0 -prec_div=1 -prec_sqrt=1 -fmad=1 " \
  "--image3=kind=elf,sm=61,file=keep/a_function.sm_61.cubin" \
  "--image3=kind=ptx,sm=61,file=keep/a_function.ptx" \
  --embedded-fatbin="keep/a_function.fatbin.c"  --device-c

# create .o file
gcc -D__CUDA_ARCH__=610 -D__CUDA_ARCH_LIST__=610 -D__NV_LEGACY_LAUNCH -c -x \
  c++  -DCUDA_DOUBLE_MATH_FUNCTIONS -Wno-psabi \
  "-I/usr/local/cuda-12.8/bin/../targets/x86_64-linux/include" \
  -m64 "keep/a_function.cudafe1.cpp" -o "a_function.o"\


nvcc -v -arch=sm_61 --device-c -keep -keep-dir=keep main.cu

# nvcc -v -arch=sm_61 -o main.out -keep -keep-dir=keep main.o a_function.o