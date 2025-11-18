// tensor.h
#ifndef TENSOR_H
#define TENSOR_H





typedef enum { DEVICE_CPU, DEVICE_CUDA } Device;

typedef struct Tensor Tensor;
typedef void (*BackwardFn)(Tensor*);


typedef struct Tensor {
    float* data;
    float* grad;
    int* shape;
    int* strides;
    int ndim;
    int size;
    int requires_grad;
    // ... whatever else you already have
    struct Tensor** parents;
    int n_parents;
    void (*backward)(struct Tensor* self);
    Device device;
} Tensor;



// Function declarations (prototypes)
void free_tensor(Tensor* tensor);

Tensor* create_tensor(float* data, const int* shape, int ndim, Device dev);

Tensor* create_tensor_autograd(float* data, const int* shape, int ndim, int requires_grad, Device dev);

const char* device_to_string(Device d);

void print_tensor_info(const Tensor* t);

int compute_size(const int* shape, int ndim);

int* broadcast_shapes(const int* a_shape, int a_ndim, const int* b_shape, int b_ndim, int* out_ndim);

void tensor_backward(Tensor* t, float* grad);

Tensor* tensor_to_cuda(const Tensor* src);

Tensor* tensor_from_cuda(const Tensor* src);


//Tensor* tensor_create(float* data, int* shape, int ndim, int requires_grad);

// void tensor_backward(Tensor* t, float* grad);
// etc...

#endif // TENSOR_H
