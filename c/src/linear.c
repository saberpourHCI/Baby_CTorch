#include"linear.h"
#include "cuda_utils.h"
#include "tensor.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>



Linear* linear_create(int in_features, int out_features, Device dev) {
    Linear* l = malloc(sizeof(Linear));

    int w_shape[2] = { in_features, out_features };
    int b_shape[1] = { out_features };

    l->W = create_empty_tensor(w_shape, 2, 1, dev);
    l->b = create_empty_tensor(b_shape, 1, 1, dev);

    // initialize W, b with small random values later (see below)
    return l;
}


Tensor* linear_forward(Linear* l, Tensor* x) {
    // x: [batch, in_features]
    Tensor* y = tensor_matmul_autograd(x, l->W);   // [batch, out_features]
    y = tensor_add_autograd(y, l->b);              // broadcast bias
    return y;
}
