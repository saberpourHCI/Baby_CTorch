#ifndef OPS_ADD_SUB
#define OPS_ADD_SUB

#include "tensor.h"

Tensor* tensor_add_cpu(const Tensor* a, const Tensor* b);

Tensor* tensor_sub_cpu(const Tensor* a, const Tensor* b);

void backward_add_cpu(Tensor* out);

void backward_sub_cpu(Tensor* out);


Tensor* tensor_add_autograd_cpu(Tensor* A, Tensor* B);

Tensor* tensor_sub_autograd_cpu(Tensor* A, Tensor* B);

#endif