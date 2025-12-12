#ifndef OPS_MUL_DIV
#define OPS_MUL_DIV

#include "tensor.h"

void backward_mul(Tensor* out);

void backward_div(Tensor* out);

Tensor* tensor_mul(const Tensor* a, const Tensor* b);

Tensor* tensor_div(const Tensor* a, const Tensor* b);

Tensor* tensor_mul_autograd(Tensor* A, Tensor* B);

Tensor* tensor_div_autograd(Tensor* A, Tensor* B);

Tensor* tensor_matmul(const Tensor* A, const Tensor* B);

Tensor* tensor_square_autograd(Tensor* A);

#endif //OPS_MUL_DIV