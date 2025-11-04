#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

typedef struct {
    float* data;
    int* strides;
    int* shape;
    int ndim;
    int size;
    char* device; // This field is unused in your code; consider removing it if not needed.
} Tensor;

Tensor* create_tensor(float* data, const int* shape, int ndim) {
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

    return tensor;
}


void free_tensor(Tensor* tensor) {
    if (!tensor) return;
    free(tensor->shape);
    free(tensor->strides);
    free(tensor);
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
}





// Computing the number of elements using the dimensions of the tensor
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

Tensor* tensor_add(const Tensor* a, const Tensor* b) {
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

    Tensor* out = create_tensor(out_data, out_shape, out_ndim);
    free(out_shape);
    return out;
}




Tensor* tensor_subtract(const Tensor* a, const Tensor* b) {
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

    Tensor* out = create_tensor(out_data, out_shape, out_ndim);
    free(out_shape);
    return out;
}






int main() {
    float data1[6] = {1, 2, 3, 4, 5, 6};
    float data2[3] = {10, 20, 30};

    int shape1[2] = {2, 3};
    int shape2[1] = {3};

    Tensor* A = create_tensor(data1, shape1, 2);
    Tensor* B = create_tensor(data2, shape2, 1);

    Tensor* C = tensor_add(A, B);
    if (!C) return 1;

    printf("Result:\n");
    for (int i = 0; i < C->size; i++) {
        printf("%.2f ", C->data[i]);
    }
    printf("\n");

    Tensor* D = tensor_add(A, B);
    if (!D) return 1;

    printf("Result:\n");
    for (int i = 0; i < D->size; i++) {
        printf("%.2f ", D->data[i]);
    }
    printf("\n");

    free_tensor(A);
    free_tensor(B);
    free(C->data); // Since tensor_add allocates data
    free_tensor(C);
    free(D->data); // Since tensor_add allocates data
    free_tensor(D);
    return 0;
}

