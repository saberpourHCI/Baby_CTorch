#include "tensor.h"
#include "cuda_utils.h"
#include "ops_add_sub.h"
#include "ops_mul_div.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stddef.h>



Tensor* MSE(Tensor* y_pred, Tensor* y_true) {
    if(y_pred->size != y_true->size) {
        printf("inside 'MSE', size mismatch between y_pred and y_true");
    }
    Tensor* diff = tensor_sub_autograd(y_pred, y_true);

    Tensor* sqr = tensor_mul_autograd(diff, diff);
    Tensor* sum = tensor_sum_autograd(sqr);
    float N_val = (float)y_pred->size;
    int N_shape[1] = {1};
    Tensor* N = create_tensor(&N_val, N_shape, 1, 0, y_pred->device);
    Tensor* mean = tensor_div_autograd(sum, N);
    return mean;
}