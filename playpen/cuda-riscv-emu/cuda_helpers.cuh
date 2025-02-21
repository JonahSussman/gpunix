#include <new> // for placement new
#include <utility>  // for std::forward


template<typename T, typename... Args>
T* cudaNewManaged(Args&&... args) {
  T* ptr;

  if (cudaMallocManaged(&ptr, sizeof(T)) != cudaSuccess)
    return nullptr;

  new (ptr) T(std::forward<Args>(args)...);
  return ptr;
}