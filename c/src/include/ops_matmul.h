#ifndef OPS_MATMUL
#define OPS_MATMUL

#include "tensor.h"
// #ifdef __cplusplus
// extern "C" {
// #endif

void backward_mul(Tensor* out);

void backward_div(Tensor* out);

Tensor* tensor_matmul(const Tensor* a, const Tensor* b);


Tensor* tensor_matmul_autograd(Tensor* A, Tensor* B);



// #ifdef __cplusplus
// }
// #endif
#endif