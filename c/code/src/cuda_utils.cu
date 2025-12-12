#include "../include/cuda_utils.h"
// #include "ops_add_sub_cuda.h"
#include "../include/tensor.h"
#include "cuda_runtime.h"

extern "C"
void where_is_float_pointer_internal(float* ptr) {
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, ptr);

    if (err != cudaSuccess) {
        // definitely not a CUDA pointer
        printf("Pointer is on CPU, the value is: %f\n", ptr[0]);

    } else if (attr.type == cudaMemoryTypeDevice) {
        float* tmp = (float*)malloc(sizeof(float));
        printf("**\n");
        cudaMemcpy(tmp, ptr, sizeof(float), cudaMemcpyDeviceToHost);//((void**)&tmp, ptr, sizeof(float));
        printf("Pointer is on GPU and ptr[0] val is: %f\n", tmp[0]);    } else if (attr.type == cudaMemoryTypeManaged) {
        printf("Pointer is managed (unified)\n");
    } else {
        printf("Pointer is on CPU\n");
    }


}


extern "C"
void where_is_float_pointer(float* ptr) {
    
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, ptr);

    if (err != cudaSuccess) {
        // definitely not a CUDA pointer
        printf("Pointer is on CPU, the value is: %f\n", ptr[0]);

    } else if (attr.type == cudaMemoryTypeDevice) {
        float* tmp = (float*)malloc(sizeof(float));
        printf("**\n");
        cudaMemcpy(tmp, ptr, sizeof(float), cudaMemcpyDeviceToHost);//((void**)&tmp, ptr, sizeof(float));
        printf("Pointer is on GPU and ptr[0] val is: %f\n", tmp[0]);
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


// extern "C"
// void where_is_float_pointer(float* ptr) {
//     cudaPointerAttributes attr;
//     cudaError_t err = cudaPointerGetAttributes(&attr, ptr);

//     if (err != cudaSuccess) {
//         // definitely not a CUDA pointer
//         printf("Pointer is on CPU\n");
//     } else if (attr.type == cudaMemoryTypeDevice) {
//         printf("Pointer is on GPU\n");
//     } else if (attr.type == cudaMemoryTypeManaged) {
//         printf("Pointer is managed (unified)\n");
//     } else {
//         printf("Pointer is on CPU\n");
//     }
// }