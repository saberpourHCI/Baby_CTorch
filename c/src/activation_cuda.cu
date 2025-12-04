#include "tensor.h"
#include "cuda_utils.h"
// #include "activation_cuda.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stddef.h>
#include "cuda_runtime.h"




__global__ void relu_kernel(const float* a,
                            float* out,
                            int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx>=size) {
        return;
    }
    out[idx] = a[idx]>0.0? a[idx]:0.0;
}

__global__ void backward_relu_kernel(const float* data_out,
                            const float* grad_out,  
                            float* a,
                            int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx>=size) {
        return;
    }
    a[idx] = data_out[idx]>0.0? grad_out[idx]:0.0;
}



extern "C"
Tensor* relu_cuda(Tensor* x) {
        if (!x) {
        printf("relu_cuda: NULL input\n");
        return NULL;
    }

    if (x->device != DEVICE_CUDA) {
        fprintf(stderr, "relu_cuda: input not on CUDA\n");
        return NULL;
    }

    Tensor* out = create_empty_tensor(x->shape, x->ndim, x->requires_grad, x->device);

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (out->size + blockSize - 1) / blockSize;

    // int blockSize = 256;
    // int numBlocks = (out->size + blockSize - 1) / blockSize;

    relu_kernel<<<numBlocks, blockSize>>>(x->data, out->data, x->size);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    // printf("FINISHED executing relu_cuda-------------------->");
    return out;
}



extern "C"
void backward_relu_cuda(Tensor* out) {
    // printf("backward_relu_cuda\n");
    if (!out) {
        printf("backward relu_cuda: NULL input\n");
        return;
    }

    if (out->device != DEVICE_CUDA) {
        fprintf(stderr, "backward_relu_cuda: input not on CUDA\n");
        return;
    }
    Tensor* x = out->parents[0];

    int blockSize = 256;
    int numBlocks = (out->size + blockSize - 1) / blockSize;

    backward_relu_kernel<<<numBlocks, blockSize>>>(out->data, out->grad, x->grad, x->size);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    // printf("FINISHED executing backward_relu_cuda-------------------->");
}


