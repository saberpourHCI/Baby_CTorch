#include "cuda_utils.h"
// #include "ops_add_sub_cuda.h"
#include "tensor.h"
#include "cuda_runtime.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stddef.h>



/*

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

*/

/*

__global__ void tensor_add_broadcast_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    const int* __restrict__ out_shape, int out_ndim,
    const int* __restrict__ a_shape,  const int* __restrict__ a_strides, int a_ndim,
    const int* __restrict__ b_shape,  const int* __restrict__ b_strides, int b_ndim,
    int out_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= out_size) return;

    int idx_a = 0;
    int idx_b = 0;
    int rem = i;

    // This mirrors your CPU broadcasting logic:
    // walk from last dimension to first, compute coord in out,
    // map to A/B indices using their shapes/strides and broadcasting rules.
    for (int d = out_ndim - 1; d >= 0; --d) {
        int coord = rem % out_shape[d];
        rem /= out_shape[d];

        // Map to A's dim/stride
        int a_dim = 1;
        int a_str = 0;
        int a_offset = d - (out_ndim - a_ndim);
        if (a_offset >= 0) {
            a_dim = a_shape[a_offset];
            a_str = a_strides[a_offset];
        }

        // Map to B's dim/stride
        int b_dim = 1;
        int b_str = 0;
        int b_offset = d - (out_ndim - b_ndim);
        if (b_offset >= 0) {
            b_dim = b_shape[b_offset];
            b_str = b_strides[b_offset];
        }

        if (a_dim != 1) {
            idx_a += coord * a_str;
        }
        if (b_dim != 1) {
            idx_b += coord * b_str;
        }
    }

    out[i] = a[idx_a] + b[idx_b];
}


*/

// __global__ void set_float_to_data_kernel(float* data, int size, float val) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if(idx < size) {
//         data[idx] = val;// a[idx] + b[idx];
//     }
// }

__global__ void tensor_mul_broadcast_kernel(const float* a, 
    const float* b, 
    float* c, 
    const int a_ndim, const int b_ndim, const int c_ndim,
    const int* a_shape, const int* b_shape, const int* c_shape,
    const int* a_strides, const int* b_strides,
    int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx>=size) {
        return;
    }
    int idx_a = 0, idx_b = 0;
    int rem = idx;
    for (int d = c_ndim - 1; d >= 0; d--) {
            int coord = rem % c_shape[d];
            rem /= c_shape[d];

            int a_dim = (d >= c_ndim - a_ndim) ? a_shape[d - (c_ndim - a_ndim)] : 1;
            int b_dim = (d >= c_ndim - b_ndim) ? b_shape[d - (c_ndim - b_ndim)] : 1;

            int a_stride = (d >= c_ndim - a_ndim) ? a_strides[d - (c_ndim - a_ndim)] : 0;
            int b_stride = (d >= c_ndim - b_ndim) ? b_strides[d - (c_ndim - b_ndim)] : 0;

            if (a_dim != 1) idx_a += coord * a_stride;
            if (b_dim != 1) idx_b += coord * b_stride;
        }

    c[idx] = a[idx_a] * b[idx_b];
}
__global__ void tensor_div_broadcast_kernel(const float* a, 
    const float* b, 
    float* c, 
    const int a_ndim, const int b_ndim, const int c_ndim,
    const int* a_shape, const int* b_shape, const int* c_shape,
    const int* a_strides, const int* b_strides,
    int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx>=size) {
        return;
    }
    int idx_a = 0, idx_b = 0;
    int rem = idx;
    for (int d = c_ndim - 1; d >= 0; d--) {
            int coord = rem % c_shape[d];
            rem /= c_shape[d];

            int a_dim = (d >= c_ndim - a_ndim) ? a_shape[d - (c_ndim - a_ndim)] : 1;
            int b_dim = (d >= c_ndim - b_ndim) ? b_shape[d - (c_ndim - b_ndim)] : 1;

            int a_stride = (d >= c_ndim - a_ndim) ? a_strides[d - (c_ndim - a_ndim)] : 0;
            int b_stride = (d >= c_ndim - b_ndim) ? b_strides[d - (c_ndim - b_ndim)] : 0;

            if (a_dim != 1) idx_a += coord * a_stride;
            if (b_dim != 1) idx_b += coord * b_stride;
        }

    c[idx] = a[idx_a] / b[idx_b];
}


__global__ void backward_mul_broadcast_kernel(
    const float* data_a,
    const float* data_b,
    const float* grad_out,
    float* grad_a,
    float* grad_b,
    const int* out_shape, int out_ndim,
    const int* a_shape,  const int* a_strides, int a_ndim,
    const int* b_shape,  const int* b_strides, int b_ndim,
    int out_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= out_size) return;

    int idx_a = 0, idx_b = 0;
    int rem = i;
    for (int d = out_ndim - 1; d >= 0; --d) {
        int coord = rem % out_shape[d];
        rem /= out_shape[d];

        int a_dim = (d >= out_ndim - a_ndim) ? a_shape[d - (out_ndim - a_ndim)] : 1;
        int b_dim = (d >= out_ndim - b_ndim) ? b_shape[d - (out_ndim - b_ndim)] : 1;

        int a_str = (d >= out_ndim - a_ndim) ? a_strides[d - (out_ndim - a_ndim)] : 0;
        int b_str = (d >= out_ndim - b_ndim) ? b_strides[d - (out_ndim - b_ndim)] : 0;

        if (a_dim != 1) idx_a += coord * a_str;
        if (b_dim != 1) idx_b += coord * b_str;
    }

    float g = grad_out[i];

    if (grad_a) atomicAdd(&grad_a[idx_a], g*data_b[idx_b]);
    if (grad_b) atomicAdd(&grad_b[idx_b], g*data_a[idx_a]);
}
__global__ void backward_div_broadcast_kernel(
    const float* data_a,
    const float* data_b,
    const float* grad_out,
    float* grad_a,
    float* grad_b,
    const int* out_shape, int out_ndim,
    const int* a_shape,  const int* a_strides, int a_ndim,
    const int* b_shape,  const int* b_strides, int b_ndim,
    int out_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= out_size) return;

    int idx_a = 0, idx_b = 0;
    int rem = i;
    for (int d = out_ndim - 1; d >= 0; --d) {
        int coord = rem % out_shape[d];
        rem /= out_shape[d];

        int a_dim = (d >= out_ndim - a_ndim) ? a_shape[d - (out_ndim - a_ndim)] : 1;
        int b_dim = (d >= out_ndim - b_ndim) ? b_shape[d - (out_ndim - b_ndim)] : 1;

        int a_str = (d >= out_ndim - a_ndim) ? a_strides[d - (out_ndim - a_ndim)] : 0;
        int b_str = (d >= out_ndim - b_ndim) ? b_strides[d - (out_ndim - b_ndim)] : 0;

        if (a_dim != 1) idx_a += coord * a_str;
        if (b_dim != 1) idx_b += coord * b_str;
    }

    // float g = grad_out[i];

    // if (grad_a) atomicAdd(&grad_a[idx_a], g);
    // if (grad_b) atomicAdd(&grad_b[idx_b], -g);
    float g    = grad_out[i];
    float aval = data_a[idx_a];
    float bval = data_b[idx_b];
    float epsilon = 0.00001;
    if (grad_a) atomicAdd(&grad_a[idx_a], g / (bval + epsilon));
    if (grad_b) atomicAdd(&grad_b[idx_b], -g * aval / (bval * bval + epsilon));
}


// extern "C"
// float* cudaMemSetFloat(float* p, int size, float val) {
//     // Launch kernel
//     int blockSize = 256;
//     int numBlocks = (size + blockSize - 1) / blockSize;

//     set_float_to_data_kernel<<<numBlocks, blockSize>>>(p, size, val);
//     return p;
// }

extern "C"
Tensor* tensor_mul_cuda(const Tensor* A, const Tensor* B) {
    if (!A || !B) {
        fprintf(stderr, "tensor_add_cuda: NULL input\n");
        return NULL;
    }

    if (A->device != DEVICE_CUDA || B->device != DEVICE_CUDA) {
        fprintf(stderr, "tensor_add_cuda: both tensors must be on CUDA\n");
        return NULL;
    }

    if (A->size != B->size) {
        fprintf(stderr, "tensor_add_cuda: size mismatch, add would broadcast!\n");
        // return NULL;
    }

    int* shape;
    int ndim;

    shape = broadcast_shapes(A->shape, A->ndim, B->shape, B->ndim, &ndim);
    // size = compute_size(shape, ndim);
    int requires_grad;
    if (A->requires_grad || B->requires_grad) {
        requires_grad = 1;
    }
    else {
        requires_grad = 0;
    }
    Tensor* out = create_empty_tensor(shape, ndim, requires_grad, DEVICE_CUDA);

    int* a_shape_device;
    int* b_shape_device; 
    int* out_shape_device;
    int* a_strides_device;
    int* b_strides_device;

    CUDA_CHECK(cudaMalloc((void**)&a_shape_device, A->ndim * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(a_shape_device, A->shape,
                          A->ndim * sizeof(int),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void**)&b_shape_device, B->ndim * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(b_shape_device, B->shape,
                          B->ndim * sizeof(int),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void**)&out_shape_device, out->ndim * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(out_shape_device, out->shape,
                          out->ndim * sizeof(int),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void**)&a_strides_device, A->ndim * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(a_strides_device, A->strides,
                          A->ndim * sizeof(int),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void**)&b_strides_device, B->ndim * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(b_strides_device, B->strides,
                          B->ndim * sizeof(int),
                          cudaMemcpyHostToDevice));

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (out->size + blockSize - 1) / blockSize;

    tensor_mul_broadcast_kernel<<<numBlocks, blockSize>>>(A->data, 
    B->data, 
    out->data, 
    A->ndim, B->ndim, out->ndim,
    a_shape_device, b_shape_device, out_shape_device,
    a_strides_device,b_strides_device,
    out->size);

    // tensor_add_kernel<<<numBlocks, blockSize>>>(A->data, B->data, out->data, out->size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(a_shape_device));
    CUDA_CHECK(cudaFree(b_shape_device));
    CUDA_CHECK(cudaFree(out_shape_device));
    CUDA_CHECK(cudaFree(a_strides_device));
    CUDA_CHECK(cudaFree(b_strides_device));
    
    // printf("FINISHED executing tesnor_add_cuda-------------------->");

    return out;
}


extern "C"
Tensor* tensor_div_cuda(const Tensor* A, const Tensor* B) {
    if (!A || !B) {
        fprintf(stderr, "tensor_sub_cuda: NULL input\n");
        return NULL;
    }

    if (A->device != DEVICE_CUDA || B->device != DEVICE_CUDA) {
        fprintf(stderr, "tensor_sub_cuda: both tensors must be on CUDA\n");
        return NULL;
    }

    // if (A->size != B->size) {
    //     fprintf(stderr, "tensor_sub_cuda: size mismatch, add would broadcast!\n");
    //     // return NULL;
    // }

    int* shape;
    int ndim;

    shape = broadcast_shapes(A->shape, A->ndim, B->shape, B->ndim, &ndim);
    // size = compute_size(shape, ndim);
    int requires_grad;
    if (A->requires_grad || B->requires_grad) {
        requires_grad = 1;
    }
    else {
        requires_grad = 0;
    }
    Tensor* out = create_empty_tensor(shape, ndim, requires_grad, DEVICE_CUDA);

    int* a_shape_device;
    int* b_shape_device; 
    int* out_shape_device;
    int* a_strides_device;
    int* b_strides_device;

    CUDA_CHECK(cudaMalloc((void**)&a_shape_device, A->ndim * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(a_shape_device, A->shape,
                          A->ndim * sizeof(int),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void**)&b_shape_device, B->ndim * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(b_shape_device, B->shape,
                          B->ndim * sizeof(int),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void**)&out_shape_device, out->ndim * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(out_shape_device, out->shape,
                          out->ndim * sizeof(int),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void**)&a_strides_device, A->ndim * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(a_strides_device, A->strides,
                          A->ndim * sizeof(int),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void**)&b_strides_device, B->ndim * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(b_strides_device, B->strides,
                          B->ndim * sizeof(int),
                          cudaMemcpyHostToDevice));

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (out->size + blockSize - 1) / blockSize;

    tensor_div_broadcast_kernel<<<numBlocks, blockSize>>>(A->data, 
    B->data, 
    out->data, 
    A->ndim, B->ndim, out->ndim,
    a_shape_device, b_shape_device, out_shape_device,
    a_strides_device,b_strides_device,
    out->size);

    // tensor_add_kernel<<<numBlocks, blockSize>>>(A->data, B->data, out->data, out->size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    // printf("FINISHED executing tesnor_add_cuda-------------------->");
    
    CUDA_CHECK(cudaFree(a_shape_device));
    CUDA_CHECK(cudaFree(b_shape_device));
    CUDA_CHECK(cudaFree(out_shape_device));
    CUDA_CHECK(cudaFree(a_strides_device));
    CUDA_CHECK(cudaFree(b_strides_device));

    return out;
}

// extern "C"
// void where_is_int_pointer(int* ptr) {
//     cudaPointerAttributes attr;
//     cudaError_t err = cudaPointerGetAttributes(&attr, ptr);

//     if (err != cudaSuccess) {
//         // definitely not a CUDA pointer
//         printf("Pointer is on CPU\n");
//     } else if (attr.type == cudaMemoryTypeDevice) {
//         printf("Pointer is on GPU\n");
//     } else if (attr.type == cudaMemoryTypeManaged) {
//         printf("Pointer is managed (unified)\n");
//     } else {
//         printf("Pointer is on CPU\n");
//     }
// }


extern "C"
void backward_mul_cuda(Tensor* out) {
    printf("mul --> ");
    // printf("backward_mul_cuda\n");
    // printf("\n\n\n\n ENTERED THE BACKWARD_ADD_CUDA\n\n\n\n");
    if(out->device != DEVICE_CUDA) {
        printf("backward_add_cuda argument not on CUDA device");
    }

    Tensor* A = out->parents[0];
    Tensor* B = out->parents[1];

    
        // Assume all on CUDA; you can add asserts:
    // A->device == DEVICE_CUDA, B->device == DEVICE_CUDA, out->device == DEVICE_CUDA

    // const int out_ndim = out->ndim;
    const int a_ndim = A->ndim;
    const int b_ndim = B->ndim;
    const int out_size = out->size;
    // printf("\n\n\n********************************\n%d %d %d %d \n***************************************\n\n\n", out_ndim, a_ndim, b_ndim, out_size);

    if (A && A->requires_grad) {
        if (!A->grad) {
            printf("ERROR: 'backward_add_cpu' tensor 'A' requires grad, but grad pointer not allocated\n");
            CUDA_CHECK(cudaMalloc((void**)&A->grad, A->size * sizeof(float)));
            CUDA_CHECK(cudaMemset(A->grad, 0, A->size * sizeof(float)));
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
            CUDA_CHECK(cudaMalloc((void**)&B->grad, B->size * sizeof(float)));
            CUDA_CHECK(cudaMemset(B->grad, 0, B->size * sizeof(float)));
            // B->grad = (float*)calloc(B->size, sizeof(float));
            // if (!B->grad) {
            //     fprintf(stderr, "backward_add: failed to allocate B->grad\n");
            //     return;
            // }
        }
    }

    int* a_shape_device;
    int* b_shape_device; 
    int* out_shape_device;
    int* a_strides_device;
    int* b_strides_device;
    // float* a_grads_device;
    // float* b_grads_device;
    // float* out_grads_device;
    // printf("A infor is: \n");
    // print_tensor_info(A);
    // printf("B infor is: \n");
    // print_tensor_info(B);
    // printf("out->shape is on ");
    // where_is_int_pointer(out->shape);
    // printf("out->shape[0]: %d out->shape[1]: %d and ndim is %d\n", out->shape[0], out->shape[1], out->ndim);
// //=========================================================================================


// printf("\n\n\n=====================================================\n");
// // 1) Basic checks
// printf("DEBUG: out_ndim = %d\n", out_ndim);
// printf("DEBUG: out->shape ptr = %p\n", (void*)out->shape);
// for (int i = 0; i < out_ndim; ++i) {
//     printf("DEBUG: out->shape[%d] = %d\n", i, out->shape[i]);
// }

// // 2) Force-synchronize and check for earlier kernel errors
// cudaError_t e = cudaDeviceSynchronize();
// if (e != cudaSuccess) {
//     fprintf(stderr, "ERROR: cudaDeviceSynchronize before malloc/copy: %s\n",
//             cudaGetErrorString(e));
//     // optionally abort here to inspect the stack
// }

// // 3) cudaGetLastError as extra check
// e = cudaGetLastError();
// if (e != cudaSuccess) {
//     fprintf(stderr, "ERROR: cudaGetLastError before malloc/copy: %s\n",
//             cudaGetErrorString(e));
// }

// // 4) Allocate and check
// cudaError_t rc = cudaMalloc((void**)&out_shape_device, out_ndim * sizeof(int));
// if (rc != cudaSuccess) {
//     fprintf(stderr, "ERROR: cudaMalloc failed: %s\n", cudaGetErrorString(rc));
// } else {
//     printf("DEBUG: out_shape_device = %p\n", (void*)out_shape_device);
// }

// // 5) Do the copy and check
// rc = cudaMemcpy(out_shape_device, out->shape, out_ndim * sizeof(int), cudaMemcpyHostToDevice);
// if (rc != cudaSuccess) {
//     fprintf(stderr, "ERROR: cudaMemcpy failed: %s\n", cudaGetErrorString(rc));
//     // optional: call cudaDeviceSynchronize() and print again, then abort
// }
// printf("\n=====================================================\n");
//============================================================================================


    

    CUDA_CHECK(cudaMalloc((void**)&out_shape_device, out->ndim * sizeof(int)));
    //     cudaError_t e = cudaGetLastError();
    // printf("Previous CUDA error: %s\n", cudaGetErrorString(e));
    // printf("out_shape_device is on ");
    // where_is_int_pointer(out_shape_device);
    CUDA_CHECK(cudaMemcpy(out_shape_device, out->shape,
                          out->ndim * sizeof(int),
                          cudaMemcpyHostToDevice));
    



    CUDA_CHECK(cudaMalloc((void**)&a_shape_device, a_ndim * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(a_shape_device, A->shape,
                          a_ndim * sizeof(int),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void**)&b_shape_device, b_ndim * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(b_shape_device, B->shape,
                          b_ndim * sizeof(int),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void**)&a_strides_device, a_ndim * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(a_strides_device, A->strides,
                          a_ndim * sizeof(int),
                          cudaMemcpyHostToDevice));


    CUDA_CHECK(cudaMalloc((void**)&b_strides_device, b_ndim * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(b_strides_device, B->strides,
                          b_ndim * sizeof(int),
                          cudaMemcpyHostToDevice));

    
    // CUDA_CHECK(cudaMalloc((void**)&out_grads_device, out->size * sizeof(float)));
    // CUDA_CHECK(cudaMemcpy(out_grads_device, out->grad,
    //                       out->size * sizeof(float),
    //                       cudaMemcpyHostToDevice));
    
    
    
    // Launch kernel
    int blockSize = 256;
    int numBlocks = (out_size + blockSize - 1) / blockSize;

    backward_mul_broadcast_kernel<<<numBlocks, blockSize>>>(
        A->data,
        B->data,
        out->grad,
        A->grad,
        B->grad,
        out_shape_device, out->ndim,
        a_shape_device, a_strides_device, A->ndim,
        b_shape_device, b_strides_device, B->ndim,
        out->size
    );
    CUDA_CHECK(cudaGetLastError());
    // printf("\npp0\n");
    CUDA_CHECK(cudaDeviceSynchronize());
    // printf("\nout->device: %s\n", device_to_string(out->device));

    // cudaPointerAttributes attr;
    // cudaError_t err = cudaPointerGetAttributes(&attr, out->grad);

    // if (err != cudaSuccess) {
    //     // definitely not a CUDA pointer
    //     printf("Pointer is on CPU\n");
    // } else if (attr.type == cudaMemoryTypeDevice) {
    //     printf("Pointer is on GPU\n");
    // } else if (attr.type == cudaMemoryTypeManaged) {
    //     printf("Pointer is managed (unified)\n");
    // } else {
    //     printf("Pointer is on CPU\n");
    // }

    

    // CUDA_CHECK(cudaMemcpy(out->grad, out_grads_device,
    //                       out->size * sizeof(float),
    //                       cudaMemcpyDeviceToDevice));

    // Free temp device arrays
    CUDA_CHECK(cudaFree(out_shape_device));
    CUDA_CHECK(cudaFree(a_shape_device));
    CUDA_CHECK(cudaFree(a_strides_device));
    CUDA_CHECK(cudaFree(b_shape_device));
    CUDA_CHECK(cudaFree(b_strides_device));
    // printf("\n\n\n executed backward_add_cuda\n\n\n");
    // printf("--");

}


extern "C"
void backward_div_cuda(Tensor* out) {
    printf("div --> ");
    // printf("backward_div_cuda\n");
    // printf("\n\n\n\n ENTERED THE BACKWARD_SUB_CUDA\n\n\n\n");
    if(out->device != DEVICE_CUDA) {
        printf("backward_sub_cuda argument not on CUDA device");
    }

    Tensor* A = out->parents[0];
    Tensor* B = out->parents[1];

        // Assume all on CUDA; you can add asserts:
    // A->device == DEVICE_CUDA, B->device == DEVICE_CUDA, out->device == DEVICE_CUDA

    const int out_ndim = out->ndim;
    const int a_ndim = A->ndim;
    const int b_ndim = B->ndim;
    const int out_size = out->size;
    // printf("\n\n\n********************************\n%d %d %d %d \n***************************************\n\n\n", out_ndim, a_ndim, b_ndim, out_size);

    if (A && A->requires_grad) {
        if (!A->grad) {
            CUDA_CHECK(cudaMalloc((void**)&A->grad, A->size * sizeof(float)));
            CUDA_CHECK(cudaMemset(A->grad, 0, A->size * sizeof(float)));
        }
    }

    if (B && B->requires_grad) {
        if (!B->grad) {
            CUDA_CHECK(cudaMalloc((void**)&B->grad, B->size * sizeof(float)));
            CUDA_CHECK(cudaMemset(B->grad, 0, B->size * sizeof(float)));
        }
    }

    int* a_shape_device;
    int* b_shape_device; 
    int* out_shape_device;
    int* a_strides_device;
    int* b_strides_device;
    // float* a_grads_device;
    // float* b_grads_device;
    // float* out_grads_device;



    CUDA_CHECK(cudaMalloc((void**)&out_shape_device, out_ndim * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(out_shape_device, out->shape,
                          out_ndim * sizeof(int),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void**)&a_shape_device, a_ndim * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(a_shape_device, A->shape,
                          a_ndim * sizeof(int),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void**)&b_shape_device, b_ndim * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(b_shape_device, B->shape,
                          b_ndim * sizeof(int),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void**)&a_strides_device, a_ndim * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(a_strides_device, A->strides,
                          a_ndim * sizeof(int),
                          cudaMemcpyHostToDevice));


    CUDA_CHECK(cudaMalloc((void**)&b_strides_device, b_ndim * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(b_strides_device, B->strides,
                          b_ndim * sizeof(int),
                          cudaMemcpyHostToDevice));

    
    // CUDA_CHECK(cudaMalloc((void**)&out_grads_device, out->size * sizeof(float)));
    // CUDA_CHECK(cudaMemcpy(out_grads_device, out->grad,
    //                       out->size * sizeof(float),
    //                       cudaMemcpyHostToDevice));
    
    
    
    // Launch kernel
    int blockSize = 256;
    int numBlocks = (out_size + blockSize - 1) / blockSize;

    backward_div_broadcast_kernel<<<numBlocks, blockSize>>>(
        A->data,
        B->data,
        out->grad,
        A->grad,
        B->grad,
        out_shape_device, out->ndim,
        a_shape_device, a_strides_device, A->ndim,
        b_shape_device, b_strides_device, B->ndim,
        out->size
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    // printf("\nout->device: %s\n", device_to_string(out->device));

    // CUDA_CHECK(cudaMemcpy(out->grad, out_grads_device,
    //                       out->size * sizeof(float),
    //                       cudaMemcpyDeviceToDevice));

    // Free temp device arrays
    CUDA_CHECK(cudaFree(out_shape_device));
    CUDA_CHECK(cudaFree(a_shape_device));
    CUDA_CHECK(cudaFree(a_strides_device));
    CUDA_CHECK(cudaFree(b_shape_device));
    CUDA_CHECK(cudaFree(b_strides_device));
    // printf("\n\n\n executed backward_add_cuda\n\n\n");

}

/*

Tensor* tensor_mul_autograd_cuda(Tensor* A, Tensor* B) {
    printf("mul --> ");
    Tensor* out = tensor_mul_cuda(A, B);
    if (!out) return NULL;

    if (out->requires_grad) {
        out->parents = (Tensor**)malloc(2 * sizeof(Tensor*));
        out->parents[0] = A;
        out->parents[1] = B;
        out->n_parents = 2;
        out->backward = backward_mul_cuda;
        CUDA_CHECK(cudaMalloc((void**)&out->grad, out->size * sizeof(float)));
        // out->grad = (float*)calloc(out->size, sizeof(float));
    }

    return out;
}

Tensor* tensor_div_autograd_cuda(Tensor* A, Tensor* B) {
    printf("div --> ");
    Tensor* out = tensor_div_cuda(A, B);
    if (!out) return NULL;

    if (A->requires_grad || B->requires_grad) {
        out->requires_grad = 1;
        out->parents = (Tensor**)malloc(2 * sizeof(Tensor*));
        out->parents[0] = A;
        out->parents[1] = B;
        out->n_parents = 2;
        out->backward = backward_div_cuda;
        // out->grad = (float*)calloc(out->size, sizeof(float));
        CUDA_CHECK(cudaMalloc((void**)&out->grad, out->size * sizeof(float)));
    }

    return out;
}

*/




