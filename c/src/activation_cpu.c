#include "tensor.h"
#include "cuda_utils.h"
// #include "activation_cuda.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stddef.h>



Tensor* relu_cpu(Tensor* x) {
    Tensor* out = create_empty_tensor(x->shape, x->ndim, x->requires_grad, x->device);
    for(int i=0; i<out->size; i++) {
        out->data[i] = x->data[i]>0 ? x->data[i] : 0.0;
    }
    return out;
}




void backward_relu_cpu(Tensor* out) {
    Tensor* a = out->parents[0];
    for(int i=0; i<out->size; i++) {
        a->grad[i] += out->data[i]>0.0? out->grad[i] : 0.0;
    }
}

