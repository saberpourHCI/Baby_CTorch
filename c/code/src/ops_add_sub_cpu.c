#include "../include/tensor.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>


Tensor* tensor_sum_cpu(const Tensor* a) {
    int shape[1] = {1};
    Tensor* out = create_empty_tensor(shape, 1, 1, DEVICE_CPU);
    float sum = 0.0;
    for(int i=0; i<a->size; i++) {
        sum += a->data[i];
    }
    out->data[0] = sum;
    return out;
}


Tensor* tensor_add_cpu(const Tensor* a, const Tensor* b) {
    int out_ndim;
    int* out_shape = broadcast_shapes(a->shape, a->ndim, b->shape, b->ndim, &out_ndim);
    if (!out_shape) {
        fprintf(stderr, "Error: Incompatible shapes for addition\n");
        return NULL;
    }
    // printf("\n\n\n\n\n\n\n\n\n\n\n\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n%d\n\n\n\n", out_ndim);

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
    
    
    int requires_grad;
    if (a->requires_grad || b->requires_grad) {
        requires_grad = 1;
    }
    else {
        requires_grad = 0;
    }

    Tensor* out = create_tensor(out_data, out_shape, out_ndim, requires_grad, a->device);
    // if(a->device==b->device){
    //     Tensor* out = create_tensor(out_data, out_shape, out_ndim, a->device);
    // }
    // else {
    //     printf("Both tensors should be on the same device");
    // }

    free(out_shape);
    return out;
}

Tensor* tensor_sub_cpu(const Tensor* a, const Tensor* b) {
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

    int requires_grad;
    if (a->requires_grad || b->requires_grad) {
        requires_grad = 1;
    }
    else {
        requires_grad = 0;
    }
    Tensor* out = create_tensor(out_data, out_shape, out_ndim, requires_grad, DEVICE_CPU);
    free(out_shape);
    return out;
}


void backward_sum_cpu(Tensor* out) {
    printf("sum --> ");
    // printf("backward_sum_cuda\n");
    Tensor* a = out->parents[0];
    // Tensor* ones = tensor_ones(a->shape, a->ndim, 1, out->device);
    float* grads = malloc(a->size * sizeof(float));
    for(int i=0; i<a->size; i++) {
        grads[i] = out->grad[i];
    }

    for(int i=0; i<a->size; i++) {
        a->grad[i] += out->grad[0];
    }
    // if(out->device==DEVICE_CUDA) {

    //     CUDA_CHECK(cudaMemcpy(a->grad, grads, a->size * sizeof(float), cudaMemcpyHostToDevice));
    // }
    // else if(out->device==DEVICE_CPU) {
        // memcpy((void**)&a->grad, grads, a->size * sizeof(float));// grads;//ones->data;
    // }
    free(grads);
    
}



void backward_add_cpu(Tensor* out) {
    Tensor* A = out->parents[0];
    Tensor* B = out->parents[1];

    if (!A && !B) return;

    int out_ndim = out->ndim;
    int* out_shape = out->shape;
    int out_size = out->size;

    int a_ndim = A ? A->ndim : 0;
    int b_ndim = B ? B->ndim : 0;

    
    if (A && A->requires_grad) {
        if (!A->grad) {
            printf("ERROR: 'backward_add_cpu' tensor 'A' requires grad, but grad pointer not allocated\n");
            // A->grad = (float*)calloc(A->size, sizeof(float));
            // if (!A->grad) {
            //     fprintf(stderr, "backward_add: failed to allocate A->grad\n");
            //     return;
            // }
        }
    }

    if (B && B->requires_grad) {
        if (!B->grad) {
            printf("ERROR: 'backward_add_cpu' tensor 'B' requires grad, but grad pointer not allocated\n");
            // B->grad = (float*)calloc(B->size, sizeof(float));
            // if (!B->grad) {
            //     fprintf(stderr, "backward_add: failed to allocate B->grad\n");
            //     return;
            // }
        }
    }

    // Loop over all elements of out
    for (int idx_out = 0; idx_out < out_size; ++idx_out) {
        int rem = idx_out;
        int idx_a = 0;
        int idx_b = 0;

        // Walk dimensions from last to first
        for (int d = out_ndim - 1; d >= 0; --d) {
            int coord = rem % out_shape[d];
            rem /= out_shape[d];

            // Map this dimension to A's dimensions
            int a_dim = 1;
            int a_stride = 0;
            if (A) {
                if (d >= out_ndim - a_ndim) {
                    int a_axis = d - (out_ndim - a_ndim);
                    a_dim = A->shape[a_axis];
                    a_stride = A->strides[a_axis];
                }
            }

            // Map this dimension to B's dimensions
            int b_dim = 1;
            int b_stride = 0;
            if (B) {
                if (d >= out_ndim - b_ndim) {
                    int b_axis = d - (out_ndim - b_ndim);
                    b_dim = B->shape[b_axis];
                    b_stride = B->strides[b_axis];
                }
            }

            // If a_dim == 1, that dimension was broadcast in A → same index along that dim
            if (A && a_dim != 1) {
                idx_a += coord * a_stride;
            }

            // Same for B
            if (B && b_dim != 1) {
                idx_b += coord * b_stride;
            }
        }

        float g = out->grad ? out->grad[idx_out] : 0.0f;

        if (A && A->requires_grad) {
            A->grad[idx_a] += g;
        }
        if (B && B->requires_grad) {
            B->grad[idx_b] += g;
        }
    }
}

void backward_sub_cpu(Tensor* out) {
    Tensor* A = out->parents[0];
    Tensor* B = out->parents[1];

    if (!A && !B) return;

    int out_ndim = out->ndim;
    int* out_shape = out->shape;
    int out_size = out->size;

    int a_ndim = A ? A->ndim : 0;
    int b_ndim = B ? B->ndim : 0;

    
    if (A && A->requires_grad) {
        if (!A->grad) {
            printf("ERROR: 'backward_sub_cpu' tensor 'A' requires grad, but grad pointer not allocated\n");
            float* tmp = (float*) malloc(A->size* sizeof(float));
            memcpy(A->grad, tmp, A->size*sizeof(float));
            free(tmp);
        }
    }

    if (B && B->requires_grad) {
        if (!B->grad) {
            printf("ERROR: 'backward_sub_cpu' tensor 'A' requires grad, but grad pointer not allocated\n");
            float* tmp = (float*) malloc(B->size* sizeof(float));
            memcpy(B->grad, tmp, B->size*sizeof(float));
            free(tmp);
        }
    }

    // Loop over all elements of out
    for (int idx_out = 0; idx_out < out_size; ++idx_out) {
        int rem = idx_out;
        int idx_a = 0;
        int idx_b = 0;

        // Walk dimensions from last to first
        for (int d = out_ndim - 1; d >= 0; --d) {
            int coord = rem % out_shape[d];
            rem /= out_shape[d];

            // Map this dimension to A's dimensions
            int a_dim = 1;
            int a_stride = 0;
            if (A) {
                if (d >= out_ndim - a_ndim) {
                    int a_axis = d - (out_ndim - a_ndim);
                    a_dim = A->shape[a_axis];
                    a_stride = A->strides[a_axis];
                }
            }

            // Map this dimension to B's dimensions
            int b_dim = 1;
            int b_stride = 0;
            if (B) {
                if (d >= out_ndim - b_ndim) {
                    int b_axis = d - (out_ndim - b_ndim);
                    b_dim = B->shape[b_axis];
                    b_stride = B->strides[b_axis];
                }
            }

            // If a_dim == 1, that dimension was broadcast in A → same index along that dim
            if (A && a_dim != 1) {
                idx_a += coord * a_stride;
            }

            // Same for B
            if (B && b_dim != 1) {
                idx_b += coord * b_stride;
            }
        }

        float g = out->grad ? out->grad[idx_out] : 0.0f;

        if (A && A->requires_grad) {
            A->grad[idx_a] += g;
        }
        if (B && B->requires_grad) {
            B->grad[idx_b] -= g;
        }
    }
}



Tensor* tensor_add_autograd_cpu(Tensor* A, Tensor* B) {
    Tensor* out = tensor_add_cpu(A, B);
    if (!out) return NULL;

    if (out->requires_grad) {
        out->parents = (Tensor**)malloc(2 * sizeof(Tensor*));
        out->parents[0] = A;
        out->parents[1] = B;
        out->n_parents = 2;
        out->backward = backward_add_cpu;
        out->grad = (float*)calloc(out->size, sizeof(float));
    }

    return out;
}

Tensor* tensor_sub_autograd_cpu(Tensor* A, Tensor* B) {
    Tensor* out = tensor_sub_cpu(A, B);
    if (!out) return NULL;

    if (out->requires_grad) {
        out->parents = (Tensor**)malloc(2 * sizeof(Tensor*));
        out->parents[0] = A;
        out->parents[1] = B;
        out->n_parents = 2;
        out->backward = backward_sub_cpu;
        // out->grad = (float*)calloc(out->size, sizeof(float));
    }

    return out;
}

