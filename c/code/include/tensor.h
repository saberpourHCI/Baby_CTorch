// tensor.h
#ifndef TENSOR_H
#define TENSOR_H

extern int in_debug;
#ifdef __cplusplus
extern "C" {
#endif





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
    int backward_visited;
    // int grad_initialized;
    struct Tensor** parents;
    int n_parents;
    void (*backward)(struct Tensor* self);
    Device device;

    int layer_id;   // which layer "owns" this tensor (-1 = no owner)
    int param_role; // 0 = none, 1 = weight, 2 = bias
} Tensor;



void free_tensor(Tensor* tensor);

Tensor* tensor_ones(const int* shape, int ndim, int requires_grad, Device dev);

Tensor* create_empty_tensor(const int* shape, int ndim, int requires_grad, Device dev);

Tensor* create_tensor(float* data, const int* shape, int ndim, int requires_grad, Device dev);


Tensor* create_tensor_autograd(float* data, const int* shape, int ndim, int requires_grad, Device dev);

const char* device_to_string(Device d);

void print_tensor_info(const Tensor* t);

int compute_size(const int* shape, int ndim);

int* broadcast_shapes(const int* a_shape, int a_ndim, const int* b_shape, int b_ndim, int* out_ndim);

void has_nan(Tensor* a);

void tensor_backward(Tensor* t, float* grad);

Tensor* tensor_to_cuda(const Tensor* src);

Tensor* tensor_to_cpu(const Tensor* src);


//Tensor* tensor_create(float* data, int* shape, int ndim, int requires_grad);

// void tensor_backward(Tensor* t, float* grad);
// etc...


#ifdef __cplusplus
}
#endif

#endif // TENSOR_H
