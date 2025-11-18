#ifndef OPS_ADD_SUB_CUDA_H
#define OPS_ADD_SUB_CUDA_H


#include "tensor.h"


#ifdef __cplusplus
extern "C" {
#endif

// Tensor* tensor_add_cuda(const Tensor* a, const Tensor* b);
Tensor* tensor_add_cuda(const Tensor* A, const Tensor* B);

Tensor* tensor_sub_gpu(const Tensor* a, const Tensor* b);

void backward_add_gpu(Tensor* out);

void backward_sub_gpu(Tensor* out);


Tensor* tensor_add_autograd_gpu(Tensor* A, Tensor* B);

Tensor* tensor_sub_autograd_gpu(Tensor* A, Tensor* B);



#ifdef __cplusplus
}
#endif
#endif