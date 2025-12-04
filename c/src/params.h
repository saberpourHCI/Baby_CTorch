// param.h
#ifndef PARAM_H
#define PARAM_H

#ifdef __cplusplus
extern "C" {
#endif

#include "tensor.h"

typedef struct {
    Tensor** params_list;
    int params_num;
    // int capacity;
} ParamSet;

void param_list_init(ParamSet* pl);
void param_list_free(ParamSet* pl);

// register a tensor as a trainable parameter
void param_list_add(ParamSet* pl, Tensor* t);

// optimizer helpers
void param_list_zero_grad(ParamSet* pl);
void param_list_sgd_step(ParamSet* pl, float lr);

#ifdef __cplusplus
}
#endif
#endif