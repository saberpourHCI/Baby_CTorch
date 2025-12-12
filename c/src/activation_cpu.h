#ifndef ACTIVATION_CPU_H
#define ACTIVATION_CPU_H

#include "tensor.h"
Tensor* relu_cpu(Tensor* x);
void backward_relu_cpu(Tensor* out);
Tensor* tanh_cpu(Tensor* x);
void backward_tanh_cpu(Tensor* out);
#endif