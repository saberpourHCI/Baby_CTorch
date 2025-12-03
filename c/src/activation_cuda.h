#ifndef ACTIVATION_CUDA_H
#define ACTIVATION_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif


#include "tensor.h"
Tensor* relu_cuda(Tensor* x);
void backward_relu_cuda(Tensor* out);


#ifdef __cplusplus
}
#endif
#endif