#include "cuda_utils.h"
// #include "ops_add_sub_cuda.h"
#include "tensor.h"
#include "cuda_runtime.h"

extern "C"
void where_is_float_pointer(float* ptr) {
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, ptr);

    if (err != cudaSuccess) {
        // definitely not a CUDA pointer
        printf("Pointer is on CPU\n");
    } else if (attr.type == cudaMemoryTypeDevice) {
        printf("Pointer is on GPU\n");
    } else if (attr.type == cudaMemoryTypeManaged) {
        printf("Pointer is managed (unified)\n");
    } else {
        printf("Pointer is on CPU\n");
    }
}

extern "C"
void where_is_int_pointer(int* ptr) {
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, ptr);

    if (err != cudaSuccess) {
        // definitely not a CUDA pointer
        printf("Pointer is on CPU\n");
    } else if (attr.type == cudaMemoryTypeDevice) {
        printf("Pointer is on GPU\n");
    } else if (attr.type == cudaMemoryTypeManaged) {
        printf("Pointer is managed (unified)\n");
    } else {
        printf("Pointer is on CPU\n");
    }
}