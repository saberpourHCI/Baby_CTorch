#ifndef OPS_ADD_SUB
#define OPS_ADD_SUB

#include "tensor.h"

Tensor* tensor_add(const Tensor* a, const Tensor* b);

Tensor* tensor_sub(const Tensor* a, const Tensor* b);

void backward_add(Tensor* out);

void backward_sub(Tensor* out);


Tensor* tensor_add_autograd(Tensor* A, Tensor* B);

Tensor* tensor_sub_autograd(Tensor* A, Tensor* B);

#endif