#ifndef OPS_ADD_SUB_CPU
#define OPS_ADD_SUB_CPU

#include "tensor.h"

Tensor* tensor_mul_cpu(const Tensor* a, const Tensor* b);

Tensor* tensor_sub_cpu(const Tensor* a, const Tensor* b);

void backward_add_cpu(Tensor* out);

void backward_sub_cpu(Tensor* out);


Tensor* tensor_add_autograd_cpu(Tensor* A, Tensor* B);

Tensor* tensor_sub_autograd_cpu(Tensor* A, Tensor* B);

#endif