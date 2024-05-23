#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

int main()
{
    // MessageBox( 0, "Blah blah...", "My Windows app!", MB_SETFOREGROUND );
    float data[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    int shape[2] = {2,3};
    Tensor* s_pointer = create_tensor(data, shape, 2);
    Tensor s = s_pointer[0];
    printf("%d\n", s.ndim);
    printf("%d\n", s_pointer->ndim);
    return 0;
}
