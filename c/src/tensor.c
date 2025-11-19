#include "cuda_utils.h"
#include "tensor.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>





void free_tensor(Tensor* tensor) {
    if (!tensor) return;
    free(tensor->shape);
    free(tensor->strides);
    free(tensor);
}

Tensor* create_tensor(float* data, const int* shape, int ndim, Device dev) {
    if (data == NULL || shape == NULL || ndim <= 0) {
        fprintf(stderr, "Invalid input parameters\n");
        return NULL;
    }

    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    if (tensor == NULL) {
        fprintf(stderr, "Failed allocating memory for tensor\n");
        return NULL;
    }

    tensor->data = data;
    tensor->shape = (int*)malloc(ndim * sizeof(int));
    if (tensor->shape == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        free(tensor);
        return NULL;
    }
    memcpy(tensor->shape, shape, ndim * sizeof(int));
    tensor->ndim = ndim;

    tensor->size = 1;
    for (int i = 0; i < ndim; i++) {
        tensor->size *= shape[i];
    }

    tensor->strides = (int*)malloc(ndim * sizeof(int));
    if (tensor->strides == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        free(tensor->shape);
        free(tensor);
        return NULL;
    }
    int stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        tensor->strides[i] = stride;
        stride *= shape[i];
    }

    if(dev == DEVICE_CUDA) {
        tensor->device = dev;
    }
    else {
        tensor->device = DEVICE_CPU;
    }
    // if(dev == DEVICE_CPU || dev == DEVICE_CUDA) {
    //     tensor->device = dev;
    // }
    // else {
    //     tensor->device = DEVICE_CPU;
    // }
    

    return tensor;
}

Tensor* create_tensor_autograd(float* data, const int* shape, int ndim, int requires_grad, Device dev) {
    Tensor* t = create_tensor(data, shape, ndim, dev);
    if (!t) return NULL;

    t->requires_grad = requires_grad;
    t->grad = NULL;
    t->parents = NULL;
    t->n_parents = 0;
    t->backward = NULL;

    if (requires_grad) {
        t->grad = (float*)calloc(t->size, sizeof(float));
        if (!t->grad) {
            free_tensor(t);
            return NULL;
        }
    }

    return t;
}


const char* device_to_string(Device d) {
    switch (d) {
        case DEVICE_CPU:  return "CPU";
        case DEVICE_CUDA: return "CUDA";
        default:       return "UNKNOWN";
    }
}


void print_tensor_info(const Tensor* t) {
    printf("Tensor: ndim=%d, size=%d\n", t->ndim, t->size);
    printf("Shape: [");
    for (int i = 0; i < t->ndim; i++) {
        printf("%d%s", t->shape[i], i == t->ndim - 1 ? "" : ", ");
    }
    printf("]\nStrides: [");
    for (int i = 0; i < t->ndim; i++) {
        printf("%d%s", t->strides[i], i == t->ndim - 1 ? "" : ", ");
    }
    printf("]\n");

    printf("Tensor data is:\n");
    if (t->device == DEVICE_CPU) {
        for (int i = 0; i < t->size; i++) {
            printf("%f ", t->data[i]);
        }
        printf("\n");
    } else if (t->device == DEVICE_CUDA) {
        // Copy to cpu first
        float* cpu_buf = (float*)malloc(t->size * sizeof(float));
        if (!cpu_buf) {
            fprintf(stderr, "Failed to allocate cpu_buf in print_tensor_data\n");
            return;
        }

        CUDA_CHECK(cudaMemcpy(cpu_buf,
                              t->data,
                              t->size * sizeof(float),
                              cudaMemcpyDeviceToHost));

        for (int i = 0; i < t->size; i++) {
            printf("%f ", cpu_buf[i]);
        }
        printf("\n");

        free(cpu_buf);
    } else {
        printf("Unknown device\n");
    }
    printf("\n");

    printf("Device is: %s\n", device_to_string(t->device));
}

int compute_size(const int* shape, int ndim) {
    int size = 1;
    for (int i = 0; i < ndim; i++)
        size *= shape[i];
    return size;
}

int* broadcast_shapes(const int* a_shape, int a_ndim, const int* b_shape, int b_ndim, int* out_ndim) {
    int ndim = (a_ndim > b_ndim) ? a_ndim : b_ndim;
    int* result_shape = (int*)malloc(ndim * sizeof(int));
    if (!result_shape) return NULL;

    for (int i = 0; i < ndim; i++) {
        int a_dim = (i >= ndim - a_ndim) ? a_shape[i - (ndim - a_ndim)] : 1;
        int b_dim = (i >= ndim - b_ndim) ? b_shape[i - (ndim - b_ndim)] : 1;

        if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
            free(result_shape);
            return NULL; // Incompatible shapes
        }
        result_shape[i] = (a_dim > b_dim) ? a_dim : b_dim;
    }

    *out_ndim = ndim;
    return result_shape;
}

void tensor_backward(Tensor* t, float* grad) {
    if (!t->grad) {
        t->grad = (float*)calloc(t->size, sizeof(float));
        for(int i =0; i < t->size; i++)
            t->grad[i] = 10.0f;
    }
 
    if (grad != NULL) {
        printf("NOT-null================> t->grad is: %f, and grad is: %f\n", t->grad[0], grad[0]);
    } else {
        printf("null================> t->grad is: %f, and grad is: %f\n", t->grad[0], *grad);
        for (int i = 0; i < t->size; i++)
            t->grad[i] = 1.0f;
    }

    if (t->backward) {
        t->backward(t); // Calculate the gradients of the parents
        for (int i = 0; i < t->n_parents; i++) {
            printf("t->grad is: %f ####################################################\n", t->grad[0]);
            tensor_backward(t->parents[i], t->grad); // recursive backward all the way to the root of the graph
        }
    }
}



Tensor* tensor_to_cuda(const Tensor* src) {
    if (!src) return NULL;

    if (src->device == DEVICE_CUDA) {
        fprintf(stderr, "tensor_to_cuda: tensor already on CUDA\n");
        return NULL;
    }


    Tensor* dst = (Tensor*)malloc(sizeof(Tensor));
    if (!dst) {
        fprintf(stderr, "tensor_to_cuda: failed to allocate Tensor\n");
        return NULL;
    }

    dst->ndim = src->ndim;
    dst->size = src->size;
    dst->requires_grad = src->requires_grad;
    dst->parents = NULL;     // not copying graph here (for now)
    dst->n_parents = 0;
    dst->backward = NULL;
    dst->device = DEVICE_CUDA;

    // Copy shape and strides to host memory
    dst->shape = (int*)malloc(dst->ndim * sizeof(int));
    dst->strides = (int*)malloc(dst->ndim * sizeof(int));
    if (!dst->shape || !dst->strides) {
        fprintf(stderr, "tensor_to_cuda: failed to allocate shape/strides\n");
        free(dst->shape);
        free(dst->strides);
        free(dst);
        return NULL;
    }
    memcpy(dst->shape, src->shape, dst->ndim * sizeof(int));
    memcpy(dst->strides, src->strides, dst->ndim * sizeof(int));

    // Allocate device memory for data
    CUDA_CHECK(cudaMalloc((void**)&dst->data, dst->size * sizeof(float)));

    // Copy data from host (src->data) to device (dst->data)
    CUDA_CHECK(cudaMemcpy(dst->data,
                          src->data,
                          dst->size * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Not handling gradients on GPU yet
    dst->grad = NULL;

    return dst;
}

Tensor* tensor_from_cuda(const Tensor* src) {

    if (!src) return NULL;

    if (src->device == DEVICE_CPU) {
        fprintf(stderr, "tensor_from_cuda: tensor already on CPU\n");
        return NULL;
    }
    
    Tensor* dst = (Tensor*)malloc(sizeof(Tensor));

    dst->ndim = src->ndim;
    dst->size = src->size;
    dst->requires_grad = NULL;//src->requires_grad;
    dst->parents = NULL;     // not copying graph here (for now)
    dst->n_parents = 0;
    dst->backward = NULL;
    dst->device = DEVICE_CPU;
    dst->grad = NULL;

    // Copy shape and strides to host memory
    dst->shape = (int*)malloc(dst->ndim * sizeof(int));
    dst->strides = (int*)malloc(dst->ndim * sizeof(int));
    // Instantiate the data pointer, when it is Null (0x0) the cudaMemcpy throws an error
    dst->data = (float*)malloc(src->size * sizeof(float));
    if (!dst->shape || !dst->strides || !dst->data) {
        fprintf(stderr, "tensor_fom_cuda: failed to allocate shape/strides\n");
        free(dst->shape);
        free(dst->strides);
        free(dst->data);
        free(dst);
        return NULL;
    }

    printf("\nq0: \n");
    print_tensor_info(src);
    CUDA_CHECK(cudaMemcpy(dst->data, src->data, src->size * sizeof(float), cudaMemcpyDeviceToHost));
    // CUDA_CHECK(cudaDeviceSynchronize());
    printf("inside tensor_from_cuda, src->size is: %d\n", src->size);
    // printf("inside tensor_from_cuda, src->data is: %f\n", src->data[0]);
    printf("inside tensor_from_cuda, dst->data is: %f\n", dst->data[0]);

    

    return dst;
}
