#include "include/linear.h"
#include "include/cuda_utils.h"
#include "include/tensor.h"
#include "include/ops_add_sub.h"
#include "include/ops_matmul.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

// call once somewhere in main: srand(time(NULL));

static float frand_uniform(float low, float high) {
    return low + (high - low) * ((float)rand() / (float)RAND_MAX);
}

Linear* linear_create(Model* model, int in_features, int out_features, Device dev) {
    Linear* l = malloc(sizeof(Linear));

    int w_shape[2] = { in_features, out_features };
    int b_shape[1] = { out_features };

    /* --------- weights W --------- */
    int size_w = compute_size(w_shape, 2);
    float* w = malloc(size_w * sizeof(float));

    // Xavier/Glorot uniform init
    float limit = sqrtf(6.0f / (in_features + out_features));
    printf("\n&&&&&&&&&&&&&&&&&&&&&&&here is the limit %f \n", limit);

    for (int i = 0; i < size_w; i++) {
        // w[i] = frand_uniform(-limit, limit);
        w[i] = frand_uniform(0, limit);
    }

    l->W = create_tensor(w, w_shape, 2, 1, dev);
    free(w);

    /* --------- biases b --------- */
    int size_b = compute_size(b_shape, 1);
    float* b = malloc(size_b * sizeof(float));

    for (int i = 0; i < size_b; i++) {
        b[i] = frand_uniform(0, limit);// 0.0f;
    }

    l->b = create_tensor(b, b_shape, 1, 1, dev);
    free(b);

    // assign layer id
    if (model) {
        l->id = model->next_layer_id++;
    } else {
        l->id = -1;
    }

    // mark tensors with layer_id and role
    l->W->layer_id  = l->id;
    l->W->param_role = 1; // weight
    l->b->layer_id  = l->id;
    l->b->param_role = 2; // bias

    

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
    // printf("here is the first linear weights: \n");
    // print_tensor_info(l->W);
    Tensor* y = tensor_matmul_autograd(x, l->W);   // [batch, out_features]
    // printf("here is the matmul in first linear output: \n");
    // print_tensor_info(y);
    y = tensor_add_autograd(y, l->b);              // broadcast bias

    printf("linear layer id: %d, add output size is: %d\n", l->id, y->size);
    // if(l->id==1 && y->size !=300) {
    //     printf("###############################\n###############################\n%d###############################\n\n\n", y->size);
    //     return NULL;
    // }
    return y;
}


void linear_free(Linear* l) {
    if (!l) return;
    free_tensor(l->W);
    free_tensor(l->b);
    free(l);
}