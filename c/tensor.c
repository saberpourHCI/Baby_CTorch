#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>


typedef struct Tensor Tensor;
typedef void (*BackwardFn)(Tensor*);

struct Tensor {
    float* data;
    float* grad;
    int* shape;
    int* strides;
    int ndim;
    int size;

    // Autograd parameters
    int requires_grad;    // 1 = track gradients
    Tensor** parents;     // Array of parent tensors
    int n_parents;        // Number of parent tensors
    BackwardFn backward;  // Function for gradients computation
};


/*typedef struct {
    float* data;
    int* strides;
    int* shape;
    int ndim;
    int size;
    char* device; // This field is unused in your code; consider removing it if not needed.
} Tensor;*/

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

Tensor* create_tensor_autograd(float* data, const int* shape, int ndim, int requires_grad) {
    Tensor* t = create_tensor(data, shape, ndim);
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


void backward_add(Tensor* out) {
    Tensor* A = out->parents[0];
    Tensor* B = out->parents[1];

    for (int i = 0; i < A->size; i++)
        A->grad[i] += out->grad[i];

    for (int i = 0; i < B->size; i++)
        B->grad[i] += out->grad[i];
}


void backward_sub(Tensor* out) {
    Tensor* A = out->parents[0];
    Tensor* B = out->parents[1];

    for (int i = 0; i < A->size; i++)
        A->grad[i] += out->grad[i];

    for (int i = 0; i < B->size; i++)
        B->grad[i] -= out->grad[i];
}


void backward_mul(Tensor* out) {
    Tensor* A = out->parents[0];
    Tensor* B = out->parents[1];

    for (int i = 0; i < A->size; i++) {
        A->grad[i] += B->data[i] * out->grad[i];
        B->grad[i] += A->data[i] * out->grad[i];
    }
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

Tensor* tensor_sub(const Tensor* a, const Tensor* b) {
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

Tensor* tensor_mul(const Tensor* a, const Tensor* b) {
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

    Tensor* out = create_tensor(out_data, out_shape, out_ndim);
    free(out_shape);
    return out;
}



Tensor* tensor_add_autograd(Tensor* A, Tensor* B) {
    Tensor* out = tensor_add(A, B);
    if (!out) return NULL;

    if (A->requires_grad || B->requires_grad) {
        out->requires_grad = 1;
        out->parents = (Tensor**)malloc(2 * sizeof(Tensor*));
        out->parents[0] = A;
        out->parents[1] = B;
        out->n_parents = 2;
        out->backward = backward_add;
        out->grad = (float*)calloc(out->size, sizeof(float));
    }

    return out;
}

Tensor* tensor_sub_autograd(Tensor* A, Tensor* B) {
    Tensor* out = tensor_sub(A, B);
    if (!out) return NULL;

    if (A->requires_grad || B->requires_grad) {
        out->requires_grad = 1;
        out->parents = (Tensor**)malloc(2 * sizeof(Tensor*));
        out->parents[0] = A;
        out->parents[1] = B;
        out->n_parents = 2;
        out->backward = backward_sub;
        out->grad = (float*)calloc(out->size, sizeof(float));
    }

    return out;
}

Tensor* tensor_mul_autograd(Tensor* A, Tensor* B){
    Tensor* out = tensor_mul(A, B);
    if (!out) return NULL;

    if (A->requires_grad || B->requires_grad) {
        out->requires_grad = 1;
        out->parents = (Tensor**)malloc(2 * sizeof(Tensor*));
        out->parents[0] = A;
        out->parents[1] = B;
        out->n_parents = 2;
        out->backward = backward_mul;
        out->grad = (float*)calloc(out->size, sizeof(float));
    }

    return out;
}



// Uses broadcasting for the last 2 dimensions and accounts for the batch multiplicartion
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

    Tensor* out = create_tensor(out_data, out_shape, out_ndim);
    free(out_shape);
    free(out_batch_shape);
    return out;
}


void tensor_backward(Tensor* t, float* grad) {
    // Initialize gradient if it doesn't have a value
    if (!t->grad) {
        t->grad = (float*)calloc(t->size, sizeof(float));
    }

    if (grad) {
        for (int i = 0; i < t->size; i++)
            t->grad[i] += grad[i];
    } else {
        for (int i = 0; i < t->size; i++)
            t->grad[i] = 1.0f;
    }

    if (t->backward) {
        t->backward(t); // Calculate the gradients of the parents
        for (int i = 0; i < t->n_parents; i++) {
            tensor_backward(t->parents[i], NULL); // recursive backward all the way to the root of the graph
        }
    }
}



int main() {
    float data1[3] = {1,2,3};
    float data2[3] = {4,5,6};

    int shape[1] = {3};
    Tensor* a = create_tensor_autograd(data1, shape, 1, 1);
    Tensor* b = create_tensor_autograd(data2, shape, 1, 1);

    Tensor* c = tensor_mul_autograd(a, b);  // c = a + b

    tensor_backward(c, NULL); // compute gradients

    printf("grad a: ");
    for (int i = 0; i < a->size; i++) printf("%f ", a->grad[i]);
    printf("\n");

    printf("grad b: ");
    for (int i = 0; i < b->size; i++) printf("%f ", b->grad[i]);
    printf("\n");

    free_tensor(a);
    free_tensor(b);
    free_tensor(c);
    return 0;
}




/*
int main() {
    printf("artin \n");
    // A: (2, 1, 2, 3)
    // B: (1, 4, 3, 2)
    // Expected output: (2, 4, 2, 2)
    float A_data[2 * 1 * 2 * 3];
    float B_data[1 * 4 * 3 * 2];

    for (int i = 0; i < 12; i++) A_data[i] = i + 1;     // Fill with 1..12
    for (int i = 0; i < 24; i++) B_data[i] = (i + 1) * 0.5;

    int A_shape[4] = {2, 1, 2, 3};
    int B_shape[4] = {1, 4, 3, 2};

    Tensor* A = create_tensor(A_data, A_shape, 4);
    Tensor* B = create_tensor(B_data, B_shape, 4);
    printf("p0");
    Tensor* C = tensor_matmul(A, B);
    printf("p1");
    if (!C) {
        printf("C not initialized!");
        return 1;
    }

    printf("Output shape: (");
    for (int i = 0; i < C->ndim; i++)
        printf("%d%s", C->shape[i], (i == C->ndim - 1) ? ")\n" : ", ");
    printf("Total elements: %d\n", C->size);

    // Print first few elements
    for (int i = 0; i < (C->size < 10 ? C->size : 10); i++)
        printf("%.2f ", C->data[i]);
    printf("...\n");

    free_tensor(A);
    free_tensor(B);
    free(C->data);
    free_tensor(C);
    return 0;
}
*/



/*int main() {
    float data1[6] = {1, 2, 3, 4, 5, 6};
    float data2[3] = {10, 20, 30};

    int shape1[2] = {2, 3};
    int shape2[1] = {3};

    Tensor* A = create_tensor(data1, shape1, 2);
    Tensor* B = create_tensor(data2, shape2, 1);

    Tensor* C = tensor_mul(A, B);
    if (!C) return 1;

    printf("Result:\n");
    for (int i = 0; i < C->size; i++) {
        printf("%.2f ", C->data[i]);
    }
    printf("\n");

    Tensor* D = tensor_sub(A, B);
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
*/
