// model.c
#include "model.h"

void model_init(Model* m) {
    param_list_init(&m->params);
}

void model_free(Model* m) {
    param_list_free(&m->params);
}

void model_register_param(Model* m, Tensor* t) {
    param_list_add(&m->params, t);
}

void model_zero_grad(Model* m) {
    param_list_zero_grad(&m->params);
}

void model_sgd_step(Model* m, float lr) {
    param_list_sgd_step(&m->params, lr);
}
