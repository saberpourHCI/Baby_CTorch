#include "include/tensor.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

void backward_square_cpu(Tensor* out) {

    Tensor* A = out->parents[0];

    for (int i = 0; i < out->size; i++) {
        int idx_a = 0, rem = i;
        for (int d = out->ndim - 1; d >= 0; d--) {
            int coord = rem % out->shape[d];
            rem /= out->shape[d];

            int a_has = (d >= out->ndim - A->ndim);

            int a_dim = a_has ? A->shape[d - (out->ndim - A->ndim)] : 1;

            int a_str = a_has ? A->strides[d - (out->ndim - A->ndim)] : 0;

            if (a_dim != 1) idx_a += coord * a_str;

        }

        float g = out->grad ? out->grad[i] : 1.0f;

        // Chain rule:
        //   dL/dA[idx_a] += dL/dout[i] * B[idx_b]
        //   dL/dB[idx_b] += dL/dout[i] * A[idx_a]
        // Note: multiple i can map to the same idx_a/idx_b (broadcast reduction),
        // so we accumulate (+=) instead of assigning.
        if (A->grad) A->grad[idx_a] += A->data[idx_a] * 2.0f*g;
    }
}



void backward_mul_cpu(Tensor* out) {

    Tensor* A = out->parents[0];
    Tensor* B = out->parents[1];

    for (int i = 0; i < out->size; i++) {
        int idx_a = 0, idx_b = 0, rem = i;
        for (int d = out->ndim - 1; d >= 0; d--) {
            int coord = rem % out->shape[d];
            rem /= out->shape[d];

            int a_has = (d >= out->ndim - A->ndim);
            int b_has = (d >= out->ndim - B->ndim);

            int a_dim = a_has ? A->shape[d - (out->ndim - A->ndim)] : 1;
            int b_dim = b_has ? B->shape[d - (out->ndim - B->ndim)] : 1;

            int a_str = a_has ? A->strides[d - (out->ndim - A->ndim)] : 0;
            int b_str = b_has ? B->strides[d - (out->ndim - B->ndim)] : 0;

            if (a_dim != 1) idx_a += coord * a_str;

            if (b_dim != 1) idx_b += coord * b_str;
        }

        float g = out->grad ? out->grad[i] : 1.0f;

        // Chain rule:
        //   dL/dA[idx_a] += dL/dout[i] * B[idx_b]
        //   dL/dB[idx_b] += dL/dout[i] * A[idx_a]
        // Note: multiple i can map to the same idx_a/idx_b (broadcast reduction),
        // so we accumulate (+=) instead of assigning.
        if (A->grad) A->grad[idx_a] += B->data[idx_b] * g;
        if (B->grad) B->grad[idx_b] += A->data[idx_a] * g;
    }
}


void backward_div_cpu(Tensor* out) {

    Tensor* A = out->parents[0];
    Tensor* B = out->parents[1];

    for (int i = 0; i < out->size; i++) {
        int idx_a = 0, idx_b = 0, rem = i;
        for (int d = out->ndim - 1; d >= 0; d--) {
            int coord = rem % out->shape[d];
            rem /= out->shape[d];

            int a_has = (d >= out->ndim - A->ndim);
            int b_has = (d >= out->ndim - B->ndim);

            int a_dim = a_has ? A->shape[d - (out->ndim - A->ndim)] : 1;
            int b_dim = b_has ? B->shape[d - (out->ndim - B->ndim)] : 1;

            int a_str = a_has ? A->strides[d - (out->ndim - A->ndim)] : 0;
            int b_str = b_has ? B->strides[d - (out->ndim - B->ndim)] : 0;

            if (a_dim != 1) idx_a += coord * a_str;

            if (b_dim != 1) idx_b += coord * b_str;
        }

        float g = out->grad ? out->grad[i] : 1.0f;
        float b = B->data[idx_b];

        printf("%f - ",b);
        printf("\n");
        if (A->grad) A->grad[idx_a] += 1.0/b * g;
        if (B->grad) B->grad[idx_b] += -(A->data[idx_a]/(b* b)) * g;
        printf("A->grad[%d] is %f \n", idx_a, A->grad[idx_a]);
    }
}



Tensor* tensor_mul_cpu(const Tensor* a, const Tensor* b) {
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

        out_data[i] = a->data[idx_a] * b->data[idx_b];
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

Tensor* tensor_div_cpu(const Tensor* a, const Tensor* b) {
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

        float denom = b->data[idx_b];
        // simple guard; you can choose to error out instead
        if (denom == 0.0f) {
            fprintf(stderr, "Warning: division by zero at element %d, setting to 0\n", i);
            out_data[i] = 0.0f;
        } else {
            out_data[i] = a->data[idx_a] / denom;
        }
    }
    int out_requires_grad = (a->requires_grad==1 && b->requires_grad==1)?1:0;
    // printf("out_requries_grad is ====> %d\n", out_requires_grad);
    Tensor* out = create_tensor_autograd(out_data, out_shape, out_ndim, out_requires_grad, DEVICE_CPU);
    free(out_shape);
    return out;
}


// Tensor* tensor_mul_autograd(Tensor* A, Tensor* B){
//     Tensor* out = tensor_mul(A, B);
//     if (!out) return NULL;

//     if (A->requires_grad || B->requires_grad) {
//         out->requires_grad = 1;
//         out->parents = (Tensor**)malloc(2 * sizeof(Tensor*));
//         out->parents[0] = A;
//         out->parents[1] = B;
//         out->n_parents = 2;
//         out->grad = (float*)calloc(out->size, sizeof(float));
//         out->backward = backward_mul;
        
//     }

//     return out;
// }

// Tensor* tensor_div_autograd(Tensor* A, Tensor* B){
//     Tensor* out = tensor_div(A, B);
//     if (!out) return NULL;

//     if (A->requires_grad || B->requires_grad) {
//         out->requires_grad = 1;
//         out->parents = (Tensor**)malloc(2 * sizeof(Tensor*));
//         out->parents[0] = A;
//         out->parents[1] = B;
//         out->n_parents = 2;
//         out->grad = (float*)calloc(out->size, sizeof(float));
//         out->backward = backward_div;
//         printf("p1: ----> B->grad[0] = %f\n", B->grad[0]);
        
        
//     }

//     return out;
// }
/*
Tensor* tensor_matmul(const Tensor* A, const Tensor* B) {
    if (A->ndim < 2 || B->ndim < 2) {
        fprintf(stderr, "Error: Matmul requires ndim >= 2\n");
        return NULL;
    }

    int A_m = A->shape[A->ndim - 2];
    int A_k = A->shape[A->ndim - 1];
    int B_k = B->shape[B->ndim - 2];
    int B_n = B->shape[B->ndim - 1];

    if (A_k != B_k) {
        fprintf(stderr, "Error: Inner dimensions must match for matmul\n");
        return NULL;
    }

    //Computing broadcasted batch shape===========================================
    int A_batch_ndim = A->ndim - 2;
    int B_batch_ndim = B->ndim - 2;
    int out_batch_ndim = (A_batch_ndim > B_batch_ndim) ? A_batch_ndim : B_batch_ndim;

    int* out_batch_shape = (int*)malloc(out_batch_ndim * sizeof(int));
    if (!out_batch_shape) {
        printf("b0");
        return NULL;
    }

    for (int i = 0; i < out_batch_ndim; i++) {
        int a_dim = (i >= out_batch_ndim - A_batch_ndim) ? A->shape[i - (out_batch_ndim - A_batch_ndim)] : 1;
        int b_dim = (i >= out_batch_ndim - B_batch_ndim) ? B->shape[i - (out_batch_ndim - B_batch_ndim)] : 1;

        if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
            fprintf(stderr, "Error: Incompatible batch dimensions for matmul\n");
            free(out_batch_shape);
            return NULL;
        }
        out_batch_shape[i] = (a_dim > b_dim) ? a_dim : b_dim;
    }

    //Create final output shape ===========================================
    int out_ndim = out_batch_ndim + 2;
    int* out_shape = (int*)malloc(out_ndim * sizeof(int));
    if (!out_shape) {
        printf("b1");
        free(out_batch_shape);
        return NULL;
    }

    memcpy(out_shape, out_batch_shape, out_batch_ndim * sizeof(int));
    out_shape[out_ndim - 2] = A_m;
    out_shape[out_ndim - 1] = B_n;

    int out_size = compute_size(out_shape, out_ndim);
    float* out_data = (float*)calloc(out_size, sizeof(float));
    if (!out_data) {
        free(out_batch_shape);
        free(out_shape);
        fprintf(stderr, "Error: Memory allocation failed\n");
        return NULL;
    }

    //Iterate over all broadcasted batches ===========================================
    int total_batches = compute_size(out_batch_shape, out_batch_ndim);

    for (int batch_idx = 0; batch_idx < total_batches; batch_idx++) {
        // Compute multi-dimensional batch index
        int rem = batch_idx;
        int a_offset = 0;
        int b_offset = 0;

        for (int d = out_batch_ndim - 1; d >= 0; d--) {
            int coord = rem % out_batch_shape[d];
            rem /= out_batch_shape[d];

            int a_dim = (d >= out_batch_ndim - A_batch_ndim) ? A->shape[d - (out_batch_ndim - A_batch_ndim)] : 1;
            int b_dim = (d >= out_batch_ndim - B_batch_ndim) ? B->shape[d - (out_batch_ndim - B_batch_ndim)] : 1;

            int a_stride = (d >= out_batch_ndim - A_batch_ndim) ? A->strides[d - (out_batch_ndim - A_batch_ndim)] : 0;
            int b_stride = (d >= out_batch_ndim - B_batch_ndim) ? B->strides[d - (out_batch_ndim - B_batch_ndim)] : 0;

            if (a_dim != 1) a_offset += coord * a_stride;
            if (b_dim != 1) b_offset += coord * b_stride;
        }

        // a_offset *= (A_m * A_k);
        // b_offset *= (B_k * B_n);
        int c_offset = batch_idx * (A_m * B_n);

        //Matrix multiplication per batch ===========================================
        for (int i = 0; i < A_m; i++) {
            for (int j = 0; j < B_n; j++) {
                float sum = 0.0f;
                for (int k = 0; k < A_k; k++) {
                    float a_val = A->data[a_offset + i * A_k + k];
                    float b_val = B->data[b_offset + k * B_n + j];
                    sum += a_val * b_val;
                }
                out_data[c_offset + i * B_n + j] = sum;
            }
        }
    }
    int requires_grad;
    if (A->requires_grad || B->requires_grad) {
        requires_grad = 1;
    }
    else {
        requires_grad = 0;
    }
    Tensor* out = create_tensor(out_data, out_shape, out_ndim, requires_grad, DEVICE_CPU);
    free(out_shape);
    free(out_batch_shape);
    return out;
}
*/