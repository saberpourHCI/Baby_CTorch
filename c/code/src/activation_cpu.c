#include "../include/tensor.h"
#include "../include/cuda_utils.h"
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






Tensor* tanh_cpu(Tensor* x){//}, Tensor* output) {
    Tensor* out = create_empty_tensor(x->shape, x->ndim, x->requires_grad, x->device);
    // assumes input->size == output->size
    for (int i = 0; i < x->size; i++) {
        float t = x->data[i];
        out->data[i] = tanhf(t);  // standard C tanh function
    }
    return out;
}

void backward_tanh_cpu(Tensor* out){//}, Tensor* output) {
    Tensor* a = out->parents[0];
    for(int i=0; i<out->size; i++) {
        float t = out->data[i];
        a->grad[i] += out->grad[i] * (1-t*t);// out->data[i]>0.0? out->grad[i] : 0.0;
    }
}