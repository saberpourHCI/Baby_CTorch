#ifndef OPS_MUL_DIV_CUDA_H
#define OPS_MUL_DIV_CUDA_H


#include "tensor.h"


#ifdef __cplusplus
extern "C" {
#endif

// Tensor* tensor_add_cuda(const Tensor* a, const Tensor* b);
Tensor* tensor_mul_cuda(const Tensor* A, const Tensor* B);

Tensor* tensor_div_cuda(const Tensor* a, const Tensor* b);

void backward_mul_cuda(Tensor* out);

void backward_div_cuda(Tensor* out);

Tensor* tensor_mul_autograd_cuda(Tensor* A, Tensor* B);

Tensor* tensor_div_autograd_cuda(Tensor* A, Tensor* B);

void backward_square_cuda(Tensor* out);

// void where_is_int_pointer(int* ptr);

#ifdef __cplusplus
}
#endif
#endif