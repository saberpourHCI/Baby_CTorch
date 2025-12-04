#include "cuda_utils.h"
#include "cuda_runtime.h"
#include "tensor.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>


__global__ void matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int m, int k, int n
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= m || col >= n) return;

    float sum = 0.0f;
    for (int p = 0; p < k; ++p) {
        sum += A[row * k + p] * B[p * n + col];
    }
    C[row * n + col] = sum;
}

// grad_a[i, p] = sum_j grad_c[i, j] * data_b[p, j]
__global__ void matmul_gradA_kernel(
    const float* __restrict__ grad_c,   // out->grad, shape (m, n)
    const float* __restrict__ data_b,   // shape (k, n)
    float* __restrict__ grad_a,     // shape (m, k)
    int m, int k, int n
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y; // row in A
    int p = blockIdx.x * blockDim.x + threadIdx.x; // col in A

    if (i >= m || p >= k) return;

    float sum = 0.0f;
    for (int j = 0; j < n; ++j) {
        float g_ij = grad_c[i * n + j];
        float b_pj = data_b[p * n + j];
        sum += g_ij * b_pj;
    }
    grad_a[i * k + p] += sum;
}

// grad_b[p, j] = sum_i data_a[i, p] * grad_c[i, j]
__global__ void matmul_gradB_kernel(
    const float* __restrict__ grad_c,   // out->grad, shape (m, n)
    const float* __restrict__ data_a,   // shape (m, k)
    float* __restrict__ grad_b,     // shape (k, n)
    int m, int k, int n
) {
    int p = blockIdx.y * blockDim.y + threadIdx.y; // row in B
    int j = blockIdx.x * blockDim.x + threadIdx.x; // col in B

    if (p >= k || j >= n) return;

    float sum = 0.0f;
    for (int i = 0; i < m; ++i) {
        float a_ip = data_a[i * k + p];
        float g_ij = grad_c[i * n + j];
        sum += a_ip * g_ij;
    }
    grad_b[p * n + j] += sum;
}



extern "C" 
Tensor* tensor_matmul_cuda(const Tensor* A, const Tensor* B) {
    if (!A || !B) {
        fprintf(stderr, "tensor_matmul_cuda: NULL input\n");
        return NULL;
    }

    if (A->device != DEVICE_CUDA || B->device != DEVICE_CUDA) {
        fprintf(stderr, "tensor_matmul_cuda: both tensors must be on CUDA\n");
        return NULL;
    }

    if (A->ndim != 2 || B->ndim != 2) {
        fprintf(stderr, "tensor_matmul_cuda: only 2D matrices supported for now\n");
        return NULL;
    }

    int m = A->shape[0];
    int kA = A->shape[1];
    // int kB = B->shape[0];
    int n = B->shape[1];

    int out_shape[2] = { m, n };

    int requires_grad;
    if(A->requires_grad==1 || B->requires_grad==1) {
        requires_grad = 1;
    }
    else {
        requires_grad = 0;
    }
    // Allocate output Tensor on host (metadata)
    Tensor* C = create_empty_tensor(out_shape, 2, requires_grad, DEVICE_CUDA); //(Tensor*)malloc(sizeof(Tensor));
    if (!C) {
        fprintf(stderr, "tensor_matmul_cuda: failed to allocate Tensor\n");
        return NULL;
    }

    // C->ndim = 2;
    // C->size = m * n;
    // C->device = DEVICE_CUDA;

    // C->parents   = NULL;
    // C->n_parents = 0;
    // C->backward  = NULL;
    // C->grad      = NULL;

    // // C->shape = (int*)malloc(2 * sizeof(int));
    // // C->strides = (int*)malloc(2 * sizeof(int));
    // if (!C->shape || !C->strides) {
    //     fprintf(stderr, "tensor_matmul_cuda: failed to allocate shape/strides\n");
    //     free(C->shape);
    //     free(C->strides);
    //     free(C);
    //     return NULL;
    // }
    // C->shape[0] = m;
    // C->shape[1] = n;
    // C->strides[1] = 1;
    // C->strides[0] = n;

    // Allocate device memory for C->data
    // CUDA_CHECK(cudaMalloc((void**)&C->data, C->size * sizeof(float)));

    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x,
              (m + block.y - 1) / block.y);

    matmul_kernel<<<grid, block>>>(
        A->data, B->data, C->data,
        m, kA, n
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return C;
}





extern "C" 
void backward_matmul_cuda(Tensor* out) {
    // printf("backward_matmul_cuda\n");
    Tensor* a = out->parents[0];
    Tensor* b = out->parents[1];

    if (!a || !b) return;
    if (!out->grad) {
        fprintf(stderr, "backward_matmul_gpu: out->grad is NULL\n");
        return;
    }

    if (a->device != DEVICE_CUDA || b->device != DEVICE_CUDA || out->device != DEVICE_CUDA) {
        fprintf(stderr, "backward_matmul_gpu: all tensors must be on CUDA\n");
        return;
    }

    if (a->ndim != 2 || b->ndim != 2 || out->ndim != 2) {
        fprintf(stderr, "backward_matmul_gpu: only 2D supported\n");
        return;
    }

    int m = a->shape[0];
    int k = a->shape[1];
    int n = b->shape[1];

    if (out->shape[0] != m || out->shape[1] != n) {
        fprintf(stderr, "backward_matmul_gpu: shape mismatch between out and parents\n");
        return;
    }



    
    if (a->requires_grad) {
        if (!a->grad) {
            printf("inside backward_matmul_cuda a->grad is not instantiated!");
        }
    }
if (b->requires_grad) {
        if (!b->grad) {
            printf("inside backward_matmul_cuda b->grad is not instantiated!");
        }
    }

    float* grad_out = out->grad;

    dim3 block(16, 16);

    // 1) grad_A = G @ B^T
    if (a->requires_grad) {
        dim3 grid_a((k + block.x - 1) / block.x,
                   (m + block.y - 1) / block.y);

        matmul_gradA_kernel<<<grid_a, block>>>(
            grad_out, b->data, a->grad,
            m, k, n
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // 2) grad_B = A^T @ G
    if (b->requires_grad) {
        dim3 grid_b((n + block.x - 1) / block.x,
                   (k + block.y - 1) / block.y);

        matmul_gradB_kernel<<<grid_b, block>>>(
            grad_out, a->data, b->grad,
            m, k, n
        );
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    // printf("\n finished backward_matmul_cuda");
}






/*

Tensor* tensor_matmul_cuda(const Tensor* a, const Tensor* b) {
     if (!a || !b) {
        fprintf(stderr, "tensor_matmul_cpu: NULL input\n");
        return NULL;
    }

    if (a->ndim != 2 || b->ndim != 2) {
        fprintf(stderr, "tensor_matmul_cpu: only 2D matrices supported for now\n");
        return NULL;
    }

    int m = a->shape[0];
    int kA = a->shape[1];
    int kB = b->shape[0];
    int n = b->shape[1];

    if (kA != kB) {
        fprintf(stderr, "tensor_matmul_cpu: shape mismatch (%d x %d) @ (%d x %d)\n",
                m, kA, kB, n);
        return NULL;
    }

    // Allocate output tensor (m x n) on cPU
    int out_shape[2] = { m, n };
    int requries_grad;
    if(a->requires_grad==1 && b->requires_grad==1) {
        requries_grad = 1;
    }
    else {
        requries_grad = 0;
    }

    // Tensor* c = create_tensor(NULL, out_shape, 2, requries_grad, DEVIcE_cPU);
    Tensor* c = create_empty_tensor(out_shape, 2, requries_grad, DEVICE_CPU);
    if (!c) {
        fprintf(stderr, "tensor_matmul_cpu: failed to create output tensor\n");
        return NULL;
    }

    // Assume create_tensor allocated c->data and c->strides correctly.
    // Use strides to support non-contiguous A/B if needed.
    int a_stride_0 = a->strides[0];
    int a_stride_1 = a->strides[1];
    int b_stride_0 = b->strides[0];
    int b_stride_1 = b->strides[1];
    int c_stride_0 = c->strides[0];
    int c_stride_1 = c->strides[1];

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int p = 0; p < kA; ++p) {
                // float a_ik = a->data[i * a_stride_0 + p * a_stride_1];
                // float b_kj = b->data[p * b_stride_0 + j * b_stride_1];
                sum += a->data[i * a_stride_0 + p * a_stride_1] * b->data[p * b_stride_0 + j * b_stride_1];// a_ik * b_kj;
            }
            c->data[i * c_stride_0 + j * c_stride_1] = sum;
        }
    }

    return c;
}
*/

