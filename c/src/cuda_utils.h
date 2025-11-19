#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H


#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>


#define CUDA_CHECK(expr) do {                                      \
    cudaError_t _err = (expr);                                    \
    if (_err != cudaSuccess) {                                    \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_err), __FILE__, __LINE__);    \
        exit(1);                                                  \
    }                                                             \
} while (0)


#endif