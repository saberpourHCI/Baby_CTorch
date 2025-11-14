
#include"tensor.h"
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

    if(dev == DEVICE_CPU || dev == DEVICE_CUDA) {
        tensor->device = dev;
    }
    else {
        tensor->device = DEVICE_CPU;
    }
    

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
        printf("null================> t->grad is: %f, and grad is: %f\n", t->grad[0], grad);
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

