#include "cuda_utils.h"
// #include "ops_add_sub_cuda.h"
#include "tensor.h"
#include "cuda_runtime.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stddef.h>



__global__ void tensor_add_kernel(const float* a, const float* b, float* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

extern "C"
Tensor* tensor_add_cuda(const Tensor* A, const Tensor* B) {
    if (!A || !B) {
        fprintf(stderr, "tensor_add_cuda: NULL input\n");
        return NULL;
    }

    if (A->device != DEVICE_CUDA || B->device != DEVICE_CUDA) {
        fprintf(stderr, "tensor_add_cuda: both tensors must be on CUDA\n");
        return NULL;
    }

    if (A->size != B->size) {
        fprintf(stderr, "tensor_add_cuda: size mismatch (no broadcasting yet)\n");
        return NULL;
    }

    // Allocate output Tensor struct on host
    Tensor* out = (Tensor*)malloc(sizeof(Tensor));
    if (!out) {
        fprintf(stderr, "tensor_add_cuda: failed to allocate Tensor\n");
        return NULL;
    }

    out->ndim = A->ndim;
    out->size = A->size;
    out->requires_grad = 0;     // autograd later
    out->parents = NULL;
    out->n_parents = 0;
    out->backward = NULL;
    out->device = DEVICE_CUDA;

    // Copy shape & strides (host-side metadata)
    out->shape = (int*)malloc(out->ndim * sizeof(int));
    out->strides = (int*)malloc(out->ndim * sizeof(int));
    if (!out->shape || !out->strides) {
        fprintf(stderr, "tensor_add_cuda: failed to allocate shape/strides\n");
        free(out->shape);
        free(out->strides);
        free(out);
        return NULL;
    }
    memcpy(out->shape, A->shape, out->ndim * sizeof(int));
    memcpy(out->strides, A->strides, out->ndim * sizeof(int));

    // Allocate GPU memory for output data
    CUDA_CHECK(cudaMalloc((void**)&out->data, out->size * sizeof(float)));
    out->grad = NULL;  // we’ll add GPU grads later

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (out->size + blockSize - 1) / blockSize;

    tensor_add_kernel<<<numBlocks, blockSize>>>(A->data, B->data, out->data, out->size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return out;
}

/*
Tensor* tensor_add_gpu(const Tensor* a, const Tensor* b) {
    int out_ndim;
    int* out_shape = broadcast_shapes(a->shape, a->ndim, b->shape, b->ndim, &out_ndim);
    if (!out_shape) {
        fprintf(stderr, "Error: Incompatible shapes for addition\n");
        return NULL;
    }

    int out_size = compute_size(out_shape, out_ndim);
    float* out_data = (float*)malloc(out_size * sizeof(float));
    if (!out_data) {
        free(out_shape);
        fprintf(stderr, "Error: Memory allocation failed\n");
        return NULL;
    }

    // Iterate through all elements using linear indexing
    for (int i = 0; i < out_size; i++) {
        // Compute multi-dimensional index
        int idx_a = 0, idx_b = 0;
        int rem = i;
        for (int d = out_ndim - 1; d >= 0; d--) {
            int coord = rem % out_shape[d];
            rem /= out_shape[d];

            int a_dim = (d >= out_ndim - a->ndim) ? a->shape[d - (out_ndim - a->ndim)] : 1;
            int b_dim = (d >= out_ndim - b->ndim) ? b->shape[d - (out_ndim - b->ndim)] : 1;

            int a_stride = (d >= out_ndim - a->ndim) ? a->strides[d - (out_ndim - a->ndim)] : 0;
            int b_stride = (d >= out_ndim - b->ndim) ? b->strides[d - (out_ndim - b->ndim)] : 0;

            if (a_dim != 1) idx_a += coord * a_stride;
            if (b_dim != 1) idx_b += coord * b_stride;
        }

        out_data[i] = a->data[idx_a] + b->data[idx_b];
    }
    
    Tensor* out = create_tensor(out_data, out_shape, out_ndim, a->device);
    // if(a->device==b->device){
    //     Tensor* out = create_tensor(out_data, out_shape, out_ndim, a->device);
    // }
    // else {
    //     printf("Both tensors should be on the same device");
    // }

    free(out_shape);
    return out;
}

*/

Tensor* tensor_sub_gpu(const Tensor* a, const Tensor* b) {
    int out_ndim;
    int* out_shape = broadcast_shapes(a->shape, a->ndim, b->shape, b->ndim, &out_ndim);
    if (!out_shape) {
        fprintf(stderr, "Error: Incompatible shapes for addition\n");
        return NULL;
    }

    int out_size = compute_size(out_shape, out_ndim);
    float* out_data = (float*)malloc(out_size * sizeof(float));
    if (!out_data) {
        free(out_shape);
        fprintf(stderr, "Error: Memory allocation failed\n");
        return NULL;
    }

    // Iterate through all elements using linear indexing
    for (int i = 0; i < out_size; i++) {
        // Compute multi-dimensional index
        int idx_a = 0, idx_b = 0;
        int rem = i;
        for (int d = out_ndim - 1; d >= 0; d--) {
            int coord = rem % out_shape[d];
            rem /= out_shape[d];

            int a_dim = (d >= out_ndim - a->ndim) ? a->shape[d - (out_ndim - a->ndim)] : 1;
            int b_dim = (d >= out_ndim - b->ndim) ? b->shape[d - (out_ndim - b->ndim)] : 1;

            int a_stride = (d >= out_ndim - a->ndim) ? a->strides[d - (out_ndim - a->ndim)] : 0;
            int b_stride = (d >= out_ndim - b->ndim) ? b->strides[d - (out_ndim - b->ndim)] : 0;

            if (a_dim != 1) idx_a += coord * a_stride;
            if (b_dim != 1) idx_b += coord * b_stride;
        }

        out_data[i] = a->data[idx_a] - b->data[idx_b];
    }

    Tensor* out = create_tensor(out_data, out_shape, out_ndim, DEVICE_CPU);
    free(out_shape);
    return out;
}



void backward_add_gpu(Tensor* out) {
    Tensor* A = out->parents[0];
    Tensor* B = out->parents[1];

    for (int i = 0; i < A->size; i++)
        A->grad[i] += out->grad[i];

    for (int i = 0; i < B->size; i++)
        B->grad[i] += out->grad[i];
}

void backward_sub_gpu(Tensor* out) {
    Tensor* A = out->parents[0];
    Tensor* B = out->parents[1];

    for (int i = 0; i < A->size; i++)
        A->grad[i] += out->grad[i];

    for (int i = 0; i < B->size; i++)
        B->grad[i] -= out->grad[i];
}


Tensor* tensor_add_autograd_gpu(Tensor* A, Tensor* B) {
    Tensor* out = tensor_add_cuda(A, B);
    if (!out) return NULL;

    if (A->requires_grad || B->requires_grad) {
        out->requires_grad = 1;
        out->parents = (Tensor**)malloc(2 * sizeof(Tensor*));
        out->parents[0] = A;
        out->parents[1] = B;
        out->n_parents = 2;
        out->backward = backward_add_gpu;
        out->grad = (float*)calloc(out->size, sizeof(float));
    }

    return out;
}

Tensor* tensor_sub_autograd_gpu(Tensor* A, Tensor* B) {
    Tensor* out = tensor_sub_gpu(A, B);
    if (!out) return NULL;

    if (A->requires_grad || B->requires_grad) {
        out->requires_grad = 1;
        out->parents = (Tensor**)malloc(2 * sizeof(Tensor*));
        out->parents[0] = A;
        out->parents[1] = B;
        out->n_parents = 2;
        out->backward = backward_sub_gpu;
        out->grad = (float*)calloc(out->size, sizeof(float));
    }

    return out;
}

