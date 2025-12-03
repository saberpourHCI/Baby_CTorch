#ifndef OPS_MUL_DIV_CPU
#define OPS_MUL_DIV_CPU

#include "tensor.h"

Tensor* tensor_mul_cpu(const Tensor* a, const Tensor* b);

Tensor* tensor_div_cpu(const Tensor* a, const Tensor* b);

void backward_mul_cpu(Tensor* out);

void backward_div_cpu(Tensor* out);


Tensor* tensor_mul_autograd_cpu(Tensor* A, Tensor* B);

Tensor* tensor_div_autograd_cpu(Tensor* A, Tensor* B);

#endif