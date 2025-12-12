#ifndef MODEL_H
#define MODEL_H

#include "params.h"

typedef struct {
    ParamSet params;
    int next_layer_id;
} Model;

void model_init(Model* m);
void model_free(Model* m);

// convenience wrappers
void model_register_param(Model* m, Tensor* t);
void model_zero_grad(Model* m);
void model_sgd_step(Model* m, float lr);

#endif
