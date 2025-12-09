#include "cuda_utils.h"
#include "cuda_runtime.h"
#include "params.h"
#include "tensor.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>


__global__ void sgd_update_kernel(float* tensor_data,
                                const float* tensor_grad,
                                float lr,
                                int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx>=size) {
        return;
    }
    tensor_data[idx] -= lr * tensor_grad[idx];
}



extern "C"
void param_list_init(ParamSet* pl) {
    pl->params_list = NULL;
    pl->params_num = 0;
    // pl->capacity = 0;
}

extern "C"
void param_list_free(ParamSet* pl) {
    free(pl->params_list);
    pl->params_list = NULL;
    pl->params_num = 0;
    // pl->capacity = 0;
}

extern "C"
void param_list_add(ParamSet* pl, Tensor* t) {
    if (!t) return;
    if (!t->requires_grad) {
        return;
    }
    Tensor** new_data = (Tensor**)realloc(pl->params_list, (pl->params_num+1)*sizeof(Tensor*));
    if (!new_data) {
        fprintf(stderr, "param_list_add: realloc failed\n");
        return;
    }
    pl->params_list = new_data;
    pl->params_list[pl->params_num] = t;
    pl->params_num++;
}

extern "C"
void param_list_zero_grad(ParamSet* pl) {
    for (int i = 0; i < pl->params_num; ++i) {
        Tensor* t = pl->params_list[i];
        if (!t || !t->grad) continue;
        t->backward_visited = 0;
        if (t->device == DEVICE_CPU) {
            memset(t->grad, 0, t->size * sizeof(float));
        } else if (t->device == DEVICE_CUDA) {
            CUDA_CHECK(cudaMemset(t->grad, 0, t->size * sizeof(float)));
        }
    }
}

extern "C"
void param_list_sgd_step(ParamSet* pl, float lr) {
    printf("param_list_sgd_step");
    for (int i = 0; i < pl->params_num; ++i) {
        Tensor* t = pl->params_list[i];
        if (!t || !t->requires_grad || !t->grad) continue;
        

        if (t->device == DEVICE_CPU) {
            for (int j = 0; j < t->size; ++j) {
                t->data[j] -= lr * t->grad[j];
            }
        } else if (t->device == DEVICE_CUDA) {
            int blockSize = 256;
            int numBlocks = (t->size + blockSize - 1) / blockSize;
            sgd_update_kernel<<<numBlocks, blockSize>>>(t->data,
                                                        t->grad, 
                                                        lr, 
                                                        t->size);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
    printf("param_list_sgd_step");
}
