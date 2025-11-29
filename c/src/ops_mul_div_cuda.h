#ifndef OPS_ADD_SUB_CUDA_H
#define OPS_ADD_SUB_CUDA_H


#include "tensor.h"


#ifdef __cplusplus
extern "C" {
#endif

// Tensor* tensor_add_cuda(const Tensor* a, const Tensor* b);
Tensor* tensor_add_cuda(const Tensor* A, const Tensor* B);

Tensor* tensor_sub_cuda(const Tensor* a, const Tensor* b);

void backward_add_cuda(Tensor* out);

void backward_sub_cuda(Tensor* out);

Tensor* tensor_add_autograd_cuda(Tensor* A, Tensor* B);

Tensor* tensor_sub_autograd_cuda(Tensor* A, Tensor* B);



#ifdef __cplusplus
}
#endif
#endif