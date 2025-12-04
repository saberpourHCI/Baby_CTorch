#include "tensor.h"
#include "cuda_utils.h"
#include "ops_matmul_cpu.h"
#include "ops_matmul_cuda.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stddef.h>


Tensor* tensor_matmul(const Tensor* a, const Tensor* b) {
    if (a->device == DEVICE_CPU && b->device == DEVICE_CPU) {
        return tensor_matmul_cpu(a, b);
    }
    else if (a->device == DEVICE_CUDA && b->device == DEVICE_CUDA) {
        return tensor_matmul_cuda(a, b);
    }
    else {
        printf("tensor_add: both tensors should be on the same device");
        return NULL;
    }
}


Tensor* tensor_matmul_autograd(Tensor* A, Tensor* B) {
    // printf("tensor_matmul_autograd called \n");
    Tensor* out = tensor_matmul(A, B);
    if (!out) return NULL;

    if (out->requires_grad) {
        // out->requires_grad = 1;
        out->parents = (Tensor**)malloc(2 * sizeof(Tensor*));
        out->parents[0] = A;
        out->parents[1] = B;
        out->n_parents = 2;
        if(out->device == DEVICE_CPU) {
            out->backward = backward_matmul_cpu;
            // out->grad = (float*)calloc(out->size, sizeof(float));
        } else if(out->device == DEVICE_CUDA) {
            out->backward = backward_matmul_cuda;
            // printf("\n\nbackward cuda is assigned!!!\n\n");
            // CUDA_CHECK(cudaMalloc((void**)&out->grad, out->size * sizeof(float)));
            
            // CUDA_CHECK(cudaMemset(out->grad, 0, out->size * sizeof(float)));
        } else {
            printf("inside tensor_mul_autograd, the device is unknown!  \n");
            return NULL;
        }

        
    }

    return out;
}




/*
Tensor* tensor_div(const Tensor* a, const Tensor* b) {
    if (a->device == DEVICE_CPU && b->device == DEVICE_CPU) {
        return tensor_div_cpu(a, b);
    }
    else if (a->device == DEVICE_CUDA && b->device == DEVICE_CUDA) {
        return tensor_div_cuda(a, b);
    }
    else {
        printf("tensor_sub: both tensors should be on the same device");
        return NULL;
    }
}

Tensor* tensor_mul_autograd(Tensor* A, Tensor* B) {
    Tensor* out = tensor_mul(A, B);
    if (!out) return NULL;

    if (out->requires_grad) {
        // out->requires_grad = 1;
        out->parents = (Tensor**)malloc(2 * sizeof(Tensor*));
        out->parents[0] = A;
        out->parents[1] = B;
        out->n_parents = 2;
        if(out->device == DEVICE_CPU) {
            out->backward = backward_mul_cpu;
            // out->grad = (float*)calloc(out->size, sizeof(float));
        } else if(out->device == DEVICE_CUDA) {
            out->backward = backward_mul_cuda;
            printf("\n\nbackward cuda is assigned!!!\n\n");
            // CUDA_CHECK(cudaMalloc((void**)&out->grad, out->size * sizeof(float)));
            
            // CUDA_CHECK(cudaMemset(out->grad, 0, out->size * sizeof(float)));
        } else {
            printf("inside tensor_mul_autograd, the device is unknown!  \n");
            return NULL;
        }

        
    }

    return out;
}

Tensor* tensor_div_autograd(Tensor* A, Tensor* B) {
    Tensor* out = tensor_div(A, B);
    if (!out) return NULL;

    if (out->requires_grad) {
        // out->requires_grad = 1;
        out->parents = (Tensor**)malloc(2 * sizeof(Tensor*));
        out->parents[0] = A;
        out->parents[1] = B;
        out->n_parents = 2;
        if(out->device == DEVICE_CPU) {
            out->backward = backward_div_cpu;
            // out->grad = (float*)calloc(out->size, sizeof(float));
        } else if(out->device == DEVICE_CUDA) {
            out->backward = backward_div_cuda;
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
*/