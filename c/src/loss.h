// loss.h
#ifndef LOSS_H
#define LOSS_H

#include"tensor.h"

#ifdef __cplusplus
extern "C" {
#endif



Tensor* MSE(Tensor* y_pred, Tensor* y_true);

#ifdef __cplusplus
}
#endif

#endif // LOSS_H
