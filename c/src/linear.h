// linear.h
#ifndef LINEAR_H
#define LINEAR_H

#include"tensor.h"

#ifdef __cplusplus
extern "C" {
#endif




// typedef enum { DEVICE_CPU, DEVICE_CUDA } Device;

typedef struct Linear Linear;
// typedef void (*BackwardFn)(Tensor*);


typedef struct Linear {
    Tensor* W;   // [in_features, out_features]
    Tensor* b;   // [out_features]
} Linear;


Linear* linear_create(int in_features, int out_features, Device dev);
Tensor* linear_forward(Linear* l, Tensor* x);




#ifdef __cplusplus
}
#endif

#endif // LINEAR_H
