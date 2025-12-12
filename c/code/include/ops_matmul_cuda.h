#ifndef OPS_MATMUL_CUDA
#define OPS_MATMUL_CUDA

#include "tensor.h"


void backward_matmul_cuda(Tensor* out);


Tensor* tensor_matmul_cuda(const Tensor* a, const Tensor* b);

#endif