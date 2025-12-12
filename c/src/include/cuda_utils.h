#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H


#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

void where_is_float_pointer(float* ptr);
// void where_is_float_pointer(const float* ptr);
void where_is_int_pointer(int* ptr);

void where_is_float_pointer_internal(float* ptr);

#define CUDA_CHECK_T(expr, tensor_ptr, float_ptr) do {                        \
    cudaError_t _err = (expr);                                      \
    if (_err != cudaSuccess) {                                      \
        fprintf(stderr, "\nCUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_err), __FILE__, __LINE__);     \
        if(!in_debug) {                                             \
            in_debug = 1;                                           \
            fprintf(stderr, "Tensor involved:\n");                     \
            where_is_float_pointer_internal(float_ptr);                          \
            print_tensor_info(tensor_ptr);                              \
            in_debug= 0;                                               \
        }                                                              \
                                                                     \
        exit(1);                                                    \
    }                                                               \
} while (0)





#define CUDA_CHECK(expr) do {                                      \
    cudaError_t _err = (expr);                                    \
    if (_err != cudaSuccess) {                                    \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_err), __FILE__, __LINE__);    \
        exit(1);                                                  \
    }                                                             \
} while (0)



#ifdef __cplusplus
}
#endif
#endif