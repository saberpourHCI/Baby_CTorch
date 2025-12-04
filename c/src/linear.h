// linear.h
#ifndef LINEAR_H
#define LINEAR_H

#include "tensor.h"
#include "params.h"
#include "model.h"

// #ifdef __cplusplus
// extern "C" {
// #endif

typedef struct Linear Linear;

typedef struct Linear {
    Tensor* W;   // [in_features, out_features]
    Tensor* b;   // [out_features]
} Linear;



Linear* linear_create(Model* model, int in_features, int out_features, Device dev);
Tensor* linear_forward(Linear* l, Tensor* x);

void linear_free(Linear* l);


#endif // LINEAR_H
