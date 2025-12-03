#ifndef OPS_MATMUL_CPU
#define OPS_MATMUL_CPU

#include "tensor.h"


void backward_matmul_cpu(Tensor* out);

Tensor* tensor_matmul_cpu(const Tensor* a, const Tensor* b);

#endif