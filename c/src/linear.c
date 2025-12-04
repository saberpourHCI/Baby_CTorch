#include "linear.h"
#include "cuda_utils.h"
#include "tensor.h"
#include "ops_add_sub.h"
#include "ops_matmul.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>



Linear* linear_create(Model* model, int in_features, int out_features, Device dev) {
    Linear* l = malloc(sizeof(Linear));

    int w_shape[2] = { in_features, out_features };
    int b_shape[1] = { out_features };

    int size_w = compute_size(w_shape, 2);
    float* d = malloc(size_w * sizeof(float));
    for(int i=0; i<size_w; i++) {
        d[i] = 0.01;
    }
    l->W = create_tensor(d, w_shape, 2, 1, dev);
    free(d);
    
    int size_b = compute_size(b_shape, 1);
    float* b = malloc(size_b * sizeof(float));
    for(int i=0; i<size_b; i++) {
        b[i] = 0.01;
    }
    l->b = create_tensor(b, b_shape, 1, 1, dev);
    free(b);

    // l->W = create_empty_tensor(w_shape, 2, 1, dev);
    // l->b = create_empty_tensor(b_shape, 1, 1, dev);


    if (model) {
        model_register_param(model, l->W);
        model_register_param(model, l->b);
    }

    // initialize W, b with small random values later (see below)
    return l;
}


Tensor* linear_forward(Linear* l, Tensor* x) {
    // printf("linear_forward_called\n");
    // x: [batch, in_features]
    Tensor* y = tensor_matmul_autograd(x, l->W);   // [batch, out_features]
    y = tensor_add_autograd(y, l->b);              // broadcast bias
    return y;
}


void linear_free(Linear* l) {
    if (!l) return;
    free_tensor(l->W);
    free_tensor(l->b);
    free(l);
}