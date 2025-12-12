#ifndef ACTIVATION_H
#define ACTIVATION_H

#include"tensor.h"

#ifdef __cplusplus
extern "C" {
#endif


// forward: y = max(0, x)
Tensor* relu_autograd(Tensor* x);

Tensor* relu(Tensor* x);

Tensor* tanh_function(Tensor* x);

Tensor* tanh_autograd(Tensor* x);


#ifdef __cplusplus
}
#endif

#endif // ACTIVATION_H
