#include "tensor.h"
#include "cuda_utils.h"
#include "ops_add_sub_cpu.h"
#include "ops_add_sub_cuda.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stddef.h>

Tensor* tensor_add(const Tensor* a, const Tensor* b) {
    if (a->device == DEVICE_CPU && b->device == DEVICE_CPU) {
        return tensor_add_cpu(a, b);
    }
    else if (a->device == DEVICE_CUDA && b->device == DEVICE_CUDA) {
        return tensor_add_cuda(a, b);
    }
    else {
        printf("tensor_add: both tensors should be on the same device");
        return NULL;
    }
}


Tensor* tensor_sub(const Tensor* a, const Tensor* b) {
    if (a->device == DEVICE_CPU && b->device == DEVICE_CPU) {
        return tensor_sub_cpu(a, b);
    }
    else if (a->device == DEVICE_CUDA && b->device == DEVICE_CUDA) {
        return tensor_sub_cuda(a, b);
    }
    else {
        printf("tensor_sub: both tensors should be on the same device");
        return NULL;
    }
}
/*
Tensor* tensor_add_old(const Tensor* a, const Tensor* b) {
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

Tensor* tensor_sub_old(const Tensor* a, const Tensor* b) {
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
*/


// void backward_add(Tensor* out) {
//     Tensor* A = out->parents[0];
//     Tensor* B = out->parents[1];

//     for (int i = 0; i < A->size; i++)
//         A->grad[i] += out->grad[i];

//     for (int i = 0; i < B->size; i++)
//         B->grad[i] += out->grad[i];
// }

// void backward_sub(Tensor* out) {
//     Tensor* A = out->parents[0];
//     Tensor* B = out->parents[1];

//     for (int i = 0; i < A->size; i++)
//         A->grad[i] += out->grad[i];

//     for (int i = 0; i < B->size; i++)
//         B->grad[i] -= out->grad[i];
// }


Tensor* tensor_add_autograd(Tensor* A, Tensor* B) {
    Tensor* out = tensor_add(A, B);
    if (!out) return NULL;

    if (out->requires_grad) {
        // out->requires_grad = 1;
        out->parents = (Tensor**)malloc(2 * sizeof(Tensor*));
        out->parents[0] = A;
        out->parents[1] = B;
        out->n_parents = 2;
        if(out->device == DEVICE_CPU) {
            out->backward = backward_add_cpu;
            // out->grad = (float*)calloc(out->size, sizeof(float));
        } else if(out->device == DEVICE_CUDA) {
            out->backward = backward_add_cuda;
            printf("\n\nbackward cuda is assigned!!!\n\n");
            // CUDA_CHECK(cudaMalloc((void**)&out->grad, out->size * sizeof(float)));
            
            // CUDA_CHECK(cudaMemset(out->grad, 0, out->size * sizeof(float)));
        } else {
            printf("inside tensor_add_autograd, the device is unknown!  \n");
            return NULL;
        }

        
    }

    return out;
}

Tensor* tensor_sub_autograd(Tensor* A, Tensor* B) {
    Tensor* out = tensor_sub(A, B);
    if (!out) return NULL;

    if (out->requires_grad) {
        // out->requires_grad = 1;
        out->parents = (Tensor**)malloc(2 * sizeof(Tensor*));
        out->parents[0] = A;
        out->parents[1] = B;
        out->n_parents = 2;
        if(out->device == DEVICE_CPU) {
            out->backward = backward_sub_cpu;
            // out->grad = (float*)calloc(out->size, sizeof(float));
        } else if(out->device == DEVICE_CUDA) {
            out->backward = backward_sub_cuda;
            printf("\n\n'tensor_sub_autograd' backward cuda is assigned!!!\n\n");
            // CUDA_CHECK(cudaMalloc((void**)&out->grad, out->size * sizeof(float)));
            
            // CUDA_CHECK(cudaMemset(out->grad, 0, out->size * sizeof(float)));
        } else {
            printf("inside tensor_sub_autograd, the device is unknown!  \n");
            return NULL;
        }

        
    }

    return out;
}


// Tensor* tensor_sub_autograd(Tensor* A, Tensor* B) {
//     Tensor* out = tensor_sub(A, B);
//     if (!out) return NULL;

//     if (A->requires_grad || B->requires_grad) {
//         out->requires_grad = 1;
//         out->parents = (Tensor**)malloc(2 * sizeof(Tensor*));
//         out->parents[0] = A;
//         out->parents[1] = B;
//         out->n_parents = 2;
//         out->backward = backward_sub;
//         out->grad = (float*)calloc(out->size, sizeof(float));
//     }

//     return out;
// }

