#include "tensor.h"
#include "cuda_utils.h"
#include "activation_cpu.h"
#include "activation_cuda.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stddef.h>

Tensor* relu(Tensor* x) {
    
    if (x->device) {
        return relu_cpu(x);
    }
    else if (x->device == DEVICE_CUDA) {
        return relu_cuda(x);
    }
    else {
        printf("tensor_add: both tensors should be on the same device");
        return NULL;
    }
}


Tensor* relu_autograd(Tensor* x) {
    Tensor* out = relu(x);// create_empty_tensor(x->shape, x->ndim, x->requires_grad, x->device);

    if (!out) return NULL;

    if (out->requires_grad) {
        // out->requires_grad = 1;
        out->parents = (Tensor**)malloc(2 * sizeof(Tensor*));
        out->parents[0] = x;
        out->n_parents = 1;
        if(out->device == DEVICE_CPU) {
            out->backward = backward_relu_cpu;
            // out->grad = (float*)calloc(out->size, sizeof(float));
        } else if(out->device == DEVICE_CUDA) {
            out->backward = backward_relu_cuda;
            printf("\n\nbackward cuda is assigned!!!\n\n");
            // CUDA_CHECK(cudaMalloc((void**)&out->grad, out->size * sizeof(float)));
            
            // CUDA_CHECK(cudaMemset(out->grad, 0, out->size * sizeof(float)));
        } else {
            printf("inside tensor_mul_autograd, the device is unknown!  \n");
            return NULL;
        }
    }
}