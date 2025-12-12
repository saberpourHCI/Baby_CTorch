#include "include/tensor.h"
#include "include/cuda_utils.h"
#include "include/ops_add_sub_cpu.h"
#include "include/ops_add_sub_cuda.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stddef.h>

Tensor* tensor_sum(const Tensor* a) {
    if (a->device == DEVICE_CPU) {
        return tensor_sum_cpu(a);
    }
    else if (a->device == DEVICE_CUDA) {
        return tensor_sum_cuda(a);
    }
    else {
        printf("tensor_add: both tensors should be on the same device");
        return NULL;
    }
}



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


Tensor* tensor_add_autograd(Tensor* A, Tensor* B) {
    printf("add -->");
    // printf("add -->");
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
            // printf("\n\nbackward cuda is assigned!!!\n\n");
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
    printf("sub -->");
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
            // printf("\n\n'tensor_sub_autograd' backward cuda is assigned!!!\n\n");
            // CUDA_CHECK(cudaMalloc((void**)&out->grad, out->size * sizeof(float)));
            
            // CUDA_CHECK(cudaMemset(out->grad, 0, out->size * sizeof(float)));
        } else {
            printf("inside tensor_sub_autograd, the device is unknown!  \n");
            return NULL;
        }

        
    }

    return out;
}



// void backward_sum_cpu(Tensor* out) {
//     printf("sum --> ");
//     // printf("backward_sum_cuda\n");
//     Tensor* a = out->parents[0];
//     // Tensor* ones = tensor_ones(a->shape, a->ndim, 1, out->device);
//     float* grads = malloc(a->size * sizeof(float));
//     for(int i=0; i<a->size; i++) {
//         grads[i] = 1;
//     }
//     if(out->device==DEVICE_CUDA) {

//         CUDA_CHECK(cudaMemcpy(a->grad, grads, a->size * sizeof(float), cudaMemcpyHostToDevice));
//     }
//     else if(out->device==DEVICE_CPU) {
//         memcpy((void**)&a->grad, grads, a->size * sizeof(float));// grads;//ones->data;
//     }
//     free(grads);
    
// }



Tensor* tensor_sum_autograd(Tensor* a) {
    printf("sum --> ");
    Tensor* out = tensor_sum(a);
    if (!out) return NULL;

    if (out->requires_grad) {
        // out->requires_grad = 1;
        out->parents = (Tensor**)malloc(2 * sizeof(Tensor*));
        out->parents[0] = a;
        // out->parents[1] = B;
        out->n_parents = 1;
        if(out->device == DEVICE_CPU) {
            out->backward = backward_sum_cpu;
            // out->grad = (float*)calloc(out->size, sizeof(float));
        } else if(out->device == DEVICE_CUDA) {
            out->backward = backward_sum_cuda;
            // printf("\n\n'tensor_sub_autograd' backward cuda is assigned!!!\n\n");
            // CUDA_CHECK(cudaMalloc((void**)&out->grad, out->size * sizeof(float)));
            
            // CUDA_CHECK(cudaMemset(out->grad, 0, out->size * sizeof(float)));
        } else {
            printf("inside tensor_sub_autograd, the device is unknown!  \n");
            return NULL;
        }
    }
    return out;
}


//     if (out->requires_grad) {
//         // out->requires_grad = 1;
//         out->parents = (Tensor**)malloc(sizeof(Tensor*));
//         out->parents[0] = a;
//         out->n_parents = 1;
//         out->backward = backward_sum;
//         // if(out->device == DEVICE_CPU) {
//         //     out->backward = backward_add_cpu;
//         //     // out->grad = (float*)calloc(out->size, sizeof(float));
//         // } else if(out->device == DEVICE_CUDA) {
//         //     out->backward = backward_add_cuda;
//         //     printf("\n\n'tensor_sum_autograd' backward cuda is assigned!!!\n\n");
//         //     // CUDA_CHECK(cudaMalloc((void**)&out->grad, out->size * sizeof(float)));
            
//         //     // CUDA_CHECK(cudaMemset(out->grad, 0, out->size * sizeof(float)));
//         // } else {
//         //     printf("inside tensor_sub_autograd, the device is unknown!  \n");
//         //     return NULL;
//         // }

//     }

//     return out;
// }


