#ifndef OPS_ADD_SUB
#define OPS_ADD_SUB

#include "tensor.h"

Tensor* tensor_add_gpu(const Tensor* a, const Tensor* b);

Tensor* tensor_sub_gpu(const Tensor* a, const Tensor* b);

void backward_add_gpu(Tensor* out);

void backward_sub_gpu(Tensor* out);


Tensor* tensor_add_autograd_gpu(Tensor* A, Tensor* B);

Tensor* tensor_sub_autograd_gpu(Tensor* A, Tensor* B);

#endif