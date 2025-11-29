#include "tensor.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>


Tensor* tensor_add_cpu(const Tensor* a, const Tensor* b) {
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

    Tensor* out = create_tensor(out_data, out_shape, out_ndim, DEVICE_CPU);
    free(out_shape);
    return out;
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
            A->grad = (float*)calloc(A->size, sizeof(float));
            if (!A->grad) {
                fprintf(stderr, "backward_add: failed to allocate A->grad\n");
                return;
            }
        }
    }

    if (B && B->requires_grad) {
        if (!B->grad) {
            B->grad = (float*)calloc(B->size, sizeof(float));
            if (!B->grad) {
                fprintf(stderr, "backward_add: failed to allocate B->grad\n");
                return;
            }
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
            A->grad = (float*)calloc(A->size, sizeof(float));
            if (!A->grad) {
                fprintf(stderr, "backward_add: failed to allocate A->grad\n");
                return;
            }
        }
    }

    if (B && B->requires_grad) {
        if (!B->grad) {
            B->grad = (float*)calloc(B->size, sizeof(float));
            if (!B->grad) {
                fprintf(stderr, "backward_add: failed to allocate B->grad\n");
                return;
            }
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



// void backward_add_cpu(Tensor* out) {
//     Tensor* A = out->parents[0];
//     Tensor* B = out->parents[1];

//     for (int i = 0; i < A->size; i++)
//         A->grad[i] += out->grad[i];

//     for (int i = 0; i < B->size; i++)
//         B->grad[i] += out->grad[i];
// }

// void backward_sub_cpu(Tensor* out) {
//     Tensor* A = out->parents[0];
//     Tensor* B = out->parents[1];

//     for (int i = 0; i < A->size; i++)
//         A->grad[i] += out->grad[i];

//     for (int i = 0; i < B->size; i++)
//         B->grad[i] -= out->grad[i];
// }


Tensor* tensor_add_autograd_cpu(Tensor* A, Tensor* B) {
    Tensor* out = tensor_add_cpu(A, B);
    if (!out) return NULL;

    if (A->requires_grad || B->requires_grad) {
        out->requires_grad = 1;
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

    if (A->requires_grad || B->requires_grad) {
        out->requires_grad = 1;
        out->parents = (Tensor**)malloc(2 * sizeof(Tensor*));
        out->parents[0] = A;
        out->parents[1] = B;
        out->n_parents = 2;
        out->backward = backward_sub_cpu;
        out->grad = (float*)calloc(out->size, sizeof(float));
    }

    return out;
}

