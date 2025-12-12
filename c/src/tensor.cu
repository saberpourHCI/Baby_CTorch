#include "cuda_utils.h"
#include "tensor.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>


static int in_debug = 0;


extern "C"
void free_tensor(Tensor* tensor) {
    if (!tensor) return;
    free(tensor->shape);
    free(tensor->strides);
    free(tensor);
}

extern "C"
Tensor* tensor_ones(const int* shape, int ndim, int requires_grad, Device dev) {
    
    int size = compute_size(shape, ndim);
    printf("inside tensor_one ndim is %d and size is %d\n", ndim, size);
    float* data = (float*)malloc(size * sizeof(float));
    for (int i =0; i<size; i++) {
        data[i] = 1;
    }
    for (int i=0; i<size; i++) {
        printf("%f\n",data[i]);
    }
    Tensor* out= create_tensor_autograd(data, shape, ndim, requires_grad, dev);
    free(data);
    return out;// create_tensor_autograd(data, shape, ndim, requires_grad, dev);
}
extern "C"
Tensor* create_empty_tensor(const int* shape, int ndim, int requires_grad, Device dev) {
    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    if (tensor == NULL) {
        fprintf(stderr, "Failed allocating memory for tensor\n");
        return NULL;
    }

    tensor->layer_id  = -1;
    tensor->param_role = 0;

    
    tensor->shape = (int*)malloc(ndim * sizeof(int));
    if (tensor->shape == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        free(tensor);
        return NULL;
    }
    memcpy(tensor->shape, shape, ndim * sizeof(int));
    tensor->ndim = ndim;

    tensor->size = 1;
    for (int i = 0; i < ndim; i++) {
        tensor->size *= shape[i];
    }

    tensor->strides = (int*)malloc(ndim * sizeof(int));
    if (tensor->strides == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        free(tensor->shape);
        free(tensor);
        return NULL;
    }
    int stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        tensor->strides[i] = stride;
        stride *= shape[i];
    }

    tensor->requires_grad = requires_grad;

    if(dev == DEVICE_CUDA) {
        tensor->device = dev;
        // Allocate device memory for data
        CUDA_CHECK(cudaMalloc((void**)&tensor->data, tensor->size * sizeof(float)));
        if(requires_grad==1) {
            CUDA_CHECK(cudaMalloc((void**)&tensor->grad, tensor->size * sizeof(float)));
            CUDA_CHECK(cudaMemset(tensor->grad, 0, tensor->size * sizeof(float)));
            // tensor->grad_initialized = 0;
        }
    }
    else if(dev == DEVICE_CPU) {
        tensor->device = DEVICE_CPU;
        tensor->data = (float*)malloc(tensor->size * sizeof(float));
        if(requires_grad==1) {
            tensor->grad = (float*)malloc(tensor->size * sizeof(float));
            for(int i=0; i<tensor->size; i++) {
                tensor->grad[i] = 0.0;
            }
            // tensor->grad_initialized = 0;
        }
    }
    else {
        printf("Warning: tensor device is neither CPU nor GPU, but backed up to CPU");//(stderr, "Warning: x is negative (%d)\n", x);
        tensor->device = DEVICE_CPU;
        tensor->data = (float*)malloc(tensor->size * sizeof(float));
        if(requires_grad==1) {
            tensor->grad = (float*)malloc(tensor->size * sizeof(float));
            // tensor->grad_initialized = 0;
        }
    }
    tensor->backward_visited = 0;
    return tensor;
}
extern "C"
Tensor* create_tensor(float* data, const int* shape, int ndim, int requires_grad, Device dev) {
    if (data == NULL || shape == NULL || ndim <= 0) {
        fprintf(stderr, "Invalid input parameters\n");
        return NULL;
    }
    /*
    // Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    // if (tensor == NULL) {
    //     fprintf(stderr, "Failed allocating memory for tensor\n");
    //     return NULL;
    // }

    
    // tensor->shape = (int*)malloc(ndim * sizeof(int));
    // if (tensor->shape == NULL) {
    //     fprintf(stderr, "Memory allocation failed\n");
    //     free(tensor);
    //     return NULL;
    // }
    // memcpy(tensor->shape, shape, ndim * sizeof(int));
    // tensor->ndim = ndim;

    // tensor->size = 1;
    // for (int i = 0; i < ndim; i++) {
    //     tensor->size *= shape[i];
    // }

    // tensor->strides = (int*)malloc(ndim * sizeof(int));
    // if (tensor->strides == NULL) {
    //     fprintf(stderr, "Memory allocation failed\n");
    //     free(tensor->shape);
    //     free(tensor);
    //     return NULL;
    // }
    // int stride = 1;
    // for (int i = ndim - 1; i >= 0; i--) {
    //     tensor->strides[i] = stride;
    //     stride *= shape[i];
    // }
    */
    Tensor* tensor = create_empty_tensor(shape, ndim, requires_grad, dev);
    if(dev == DEVICE_CUDA) {
        CUDA_CHECK(cudaMemcpy(tensor->data,
                            data,
                            tensor->size * sizeof(float),
                            cudaMemcpyHostToDevice));
    }
    else {
        // tensor->data = memcpy((void*)tensor->data, data, tensor->size * sizeof(float));// data;
        memcpy(tensor->data, data, tensor->size * sizeof(float));

    }
    return tensor;
}



extern "C"
Tensor* create_tensor_autograd(float* data, const int* shape, int ndim, int requires_grad, Device dev) {
    Tensor* t = create_tensor(data, shape, ndim, requires_grad, dev);
    if (!t) return NULL;

    // t->requires_grad = requires_grad;
    // t->grad = NULL;
    t->parents = NULL;
    t->n_parents = 0;
    t->backward = NULL;

    // if (requires_grad) {
    //     if(dev == DEVICE_CPU) {
    //         t->grad = (float*)calloc(t->size, sizeof(float));
    //     }
    //     else if(dev == DEVICE_CUDA) {
    //         CUDA_CHECK(cudaMalloc((void**)&t->grad, t->size * sizeof(float)));
    //         CUDA_CHECK(cudaMemset(t->grad, 0, t->size * sizeof(float)));

    //         // CUDA_CHECK(cudaMemcpy(t->grad,
    //         //                 0,
    //         //                 t->size * sizeof(float),
    //         //                 cudaMemcpyHostToDevice));

    //     }
    //     if (!t->grad) {
    //         free_tensor(t);
    //         return NULL;
    //     }
    // }

    return t;
}

extern "C"
const char* device_to_string(Device d) {
    switch (d) {
        case DEVICE_CPU:  return "CPU";
        case DEVICE_CUDA: return "CUDA";
        default:       return "UNKNOWN";
    }
}
extern "C"
void print_tensor_info(const Tensor* t) {
    if(!t) {
        printf("\n\nprint info not possible as tensor t is NULL!!\n\n");
    }
    printf("Device is: %s\n", device_to_string(t->device));
    printf("Layer_id is: %d\n", t->layer_id);
    printf("param_role: %d\n", t->param_role);
    printf("Tensor size is: %d\n", t->size);
    printf("\nTensor: ndim=%d, size=%d\n", t->ndim, t->size);
    printf("Shape: [");
    for (int i = 0; i < t->ndim; i++) {
        printf("%d%s", t->shape[i], i == t->ndim - 1 ? "" : ", ");
    }
    printf("]\nStrides: [");
    for (int i = 0; i < t->ndim; i++) {
        printf("%d%s", t->strides[i], i == t->ndim - 1 ? "" : ", ");
    }
    printf("]\n");


    if (t->device == DEVICE_CPU) {
        printf("Tensor data is:\n");
        for (int i = 0; i < t->size; i++) {
            printf("%f ", t->data[i]);
        }
        printf("\n");


        printf("Tensor grad is:\n");
        if(t->requires_grad==1) {
            for (int i = 0; i < t->size; i++) {
                printf("%f ", t->grad[i]);
            }
            printf("\n");
                
        } else {
            printf("Tensor doesn't require grad! \n");
        }
    } else if (t->device == DEVICE_CUDA) {
        // Copy to cpu first
        float* cpu_buf = (float*)malloc(t->size * sizeof(float));
        if (!cpu_buf) {
            printf("\nFailed to allocate cpu_buf in print_tensor_data\n");
            return;
        }
        if (!t->data) {
            printf("t->data is NULL\n");
            return;
        }
        if (t->size <= 0) {
            printf("t->size is invalid: %d\n", t->size);
            return;
        }


        

        CUDA_CHECK_T(cudaMemcpy(cpu_buf,
                              t->data,
                              t->size * sizeof(float),
                              cudaMemcpyDeviceToHost), t, t->data);

        printf("Tensor data is:\n");
        for (int i = 0; i < t->size; i++) {
            printf("%f ", cpu_buf[i]);
        }
        printf("\n");

        
        free(cpu_buf);
        cpu_buf = (float*)malloc(t->size * sizeof(float));
        if(t->requires_grad == 1) {
            if (!t->grad) {
                printf("\nFailed to allocate t->grad in print_tensor_data\n");
                return;
            }
            CUDA_CHECK(cudaMemcpy(cpu_buf,
                              t->grad,
                              t->size * sizeof(float),
                              cudaMemcpyDeviceToHost));

            printf("Tensor grad is:\n");
            for (int i = 0; i < t->size; i++) {
                printf("%f ", cpu_buf[i]);
            }
            printf("\n");
        } else {
            printf("Tensor doesn't require grad! \n");
        }

        free(cpu_buf);
    } else {
        printf("Unknown device\n");
    }

    
    

    printf("==================================\n");
}
extern "C"
int compute_size(const int* shape, int ndim) {
    int size = 1;
    for (int i = 0; i < ndim; i++)
        size *= shape[i];
    return size;
}
extern "C"
int* broadcast_shapes(const int* a_shape, int a_ndim, const int* b_shape, int b_ndim, int* out_ndim) {
    int ndim = (a_ndim > b_ndim) ? a_ndim : b_ndim;
    int* result_shape = (int*)malloc(ndim * sizeof(int));
    if (!result_shape) return NULL;

    for (int i = 0; i < ndim; i++) {
        int a_dim = (i >= ndim - a_ndim) ? a_shape[i - (ndim - a_ndim)] : 1;
        int b_dim = (i >= ndim - b_ndim) ? b_shape[i - (ndim - b_ndim)] : 1;

        if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
            free(result_shape);
            return NULL; // Incompatible shapes
        }
        result_shape[i] = (a_dim > b_dim) ? a_dim : b_dim;
    }

    *out_ndim = ndim;
    return result_shape;
}

extern "C"
void has_nan(Tensor* a) {
    if(a->device==DEVICE_CUDA) {
        float* t = (float*)malloc(a->size * sizeof(float));
        CUDA_CHECK(cudaMemcpy((void*)t, 
                                a->data, 
                                a->size*sizeof(float), 
                                cudaMemcpyDeviceToHost));
        printf("\n=================================");
        for(int i=0; i<a->size; i++) {
            if (isnan(t[i])) {
                printf("found NaN %d\n", a->size);
             }
        }
        printf("=================================");

    }
    else if(a->device==DEVICE_CPU) {
        printf("=================================");
        for(int i=0; i<a->size; i++) {
            if (isnan(a->data[i])) {
                printf("found NaN %d\n", a->size);
             }
        }
        printf("=================================");

    }
}

extern "C"
void tensor_backward(Tensor* t, float* grad) {
    if(!t) return;
    if(!t->requires_grad) return;
    if(t->backward_visited ==1) {
        return;
    }
    t->backward_visited = 1;
    if (grad != NULL) {

        if(t->device == DEVICE_CUDA) {
            // --- FIX 1: Always allocate grad safely ---
            if(!t->grad) {
                CUDA_CHECK(cudaMalloc((void**)&t->grad, t->size * sizeof(float)));
            }

            // --- FIX 2: Detect pointer type to choose correct memcpy kind ---
            cudaPointerAttributes attr;
            cudaError_t err = cudaPointerGetAttributes(&attr, grad);

            cudaMemcpyKind kind;
            if (err == cudaSuccess && attr.type == cudaMemoryTypeDevice) {
                kind = cudaMemcpyDeviceToDevice;
            } else {
                kind = cudaMemcpyHostToDevice;
            }

            CUDA_CHECK_T(cudaMemcpy(t->grad, grad, t->size * sizeof(float), kind), t, grad);
        }
        else if(t->device == DEVICE_CPU) {
            // --- FIX 3: Allocate t->grad if missing on CPU ---
            if(!t->grad) {
                t->grad = (float*)malloc(t->size * sizeof(float));
            }
            memcpy(t->grad, grad, t->size * sizeof(float));
        }

    } else {
        // grad == NULL → init with 1s

        // --- FIX 4: make sure t->grad exists first ---
        if(!t->grad) {
            if(t->device == DEVICE_CUDA) {
                CUDA_CHECK_T(cudaMalloc((void**)&t->grad, t->size * sizeof(float)), t, NULL);
            } else {
                t->grad = (float*)malloc(t->size * sizeof(float));
            }
        }

        float* init_grad = (float*)malloc(t->size * sizeof(float));
        for(int i = 0; i < t->size; i++) {
            init_grad[i] = 1.0f;
        }

        if(t->device == DEVICE_CUDA) {
            CUDA_CHECK_T(cudaMemcpy(t->grad, init_grad, t->size * sizeof(float), cudaMemcpyHostToDevice), t, init_grad);
        }
        else if(t->device == DEVICE_CPU) {
            memcpy(t->grad, init_grad, t->size * sizeof(float));
        }

        free(init_grad);
    }

    // --- Backprop to parents ---
    if (t->n_parents != 0) {
        t->backward(t);

        for (int i = 0; i < t->n_parents; i++) {
            tensor_backward(t->parents[i], t->parents[i]->grad);
        }
    }
}



extern "C"
void tensor_backward_originial(Tensor* t, float* grad) {
    // printf("entred_tensor_backward\n");
    if(!t) {
        return;
    }
    if(!t->requires_grad) {
        return;
    }

    // printf("\ninside tensor_backward reached\n");  
    if (grad != NULL) {
        // printf("NOT-null================> t->grad is: something");// %f, and grad is: %f\n", t->grad[0], grad[0]);
        if(t->device == DEVICE_CUDA) {
            if(!t->grad) {
                CUDA_CHECK(cudaMalloc((void**)&t->grad, t->size * sizeof(float)));
            }
            // CUDA_CHECK(cudaMemcpy(t->grad, grad, t->size * sizeof(float), cudaMemcpyHostToDevice));
            printf("\nhere is the grad before error happens: \n");
            where_is_float_pointer(grad);
            printf("---------------------------\n");
            CUDA_CHECK_T(cudaMemcpy(t->grad, grad, t->size * sizeof(float), cudaMemcpyDeviceToDevice), t, grad);
        }
        else if(t->device == DEVICE_CPU) {
            memcpy((void*)t->grad, grad, t->size*sizeof(float));// init_grad;
        }
    } else {
        // printf("\n\n\n\n\nInside tensor_backward is reached\n\n\n\n\n");
        float* init_grad = (float*)malloc(t->size * sizeof(float));
        for(int i=0; i<t->size; i++) {
            init_grad[i] = 1.0f;
        }
        if(t->device == DEVICE_CUDA) {
            CUDA_CHECK(cudaMemcpy(t->grad, init_grad, t->size * sizeof(float), cudaMemcpyHostToDevice));
        }
        else if(t->device == DEVICE_CPU) {
            memcpy((void*)t->grad, init_grad, t->size*sizeof(float));// init_grad;
        }
        free(init_grad);
    }

    
    if (t->n_parents!=0) {
        t->backward(t); // Calculate the gradients of the parents
        for (int i = 0; i < t->n_parents; i++) {
            // Tensor* p = t->parents[i];
            // Tensor* grad_tmp;            
            // CUDA_CHECK(cudaMalloc((void**) &grad_tmp, p->size * sizeof(float)));
            // CUDA_CHECK(cudaMemcpy(grad_tmp, p->grad, p->size*sizeof(float, cuda)))
            tensor_backward(t->parents[i], t->parents[i]->grad); // recursive backward all the way to the root of the graph
            
        }
    }
}




extern "C"
void tensor_backward_backup(Tensor* t, float* grad) {
    // printf("entred_tensor_backward\n");
    if(!t) {
        return;
    }
    if(!t->requires_grad) {
        return;
    }

    // printf("\ninside tensor_backward reached\n");  
    if (grad != NULL) {
        // printf("NOT-null================> t->grad is: something");// %f, and grad is: %f\n", t->grad[0], grad[0]);
        if(t->device == DEVICE_CUDA) {
            if(!t->grad) {
                CUDA_CHECK(cudaMalloc((void**)&t->grad, t->size * sizeof(float)));
            }
            // CUDA_CHECK(cudaMemcpy(t->grad, grad, t->size * sizeof(float), cudaMemcpyHostToDevice));
            printf("\nhere is the grad before error happens: \n");
            where_is_float_pointer(grad);
            printf("---------------------------\n");
            CUDA_CHECK_T(cudaMemcpy(t->grad, grad, t->size * sizeof(float), cudaMemcpyDeviceToDevice), t, grad);
        }
        else if(t->device == DEVICE_CPU) {
            memcpy((void*)t->grad, grad, t->size*sizeof(float));// init_grad;
        }
    } else {
        // printf("\n\n\n\n\nInside tensor_backward is reached\n\n\n\n\n");
        float* init_grad = (float*)malloc(t->size * sizeof(float));
        for(int i=0; i<t->size; i++) {
            init_grad[i] = 1.0f;
        }
        if(t->device == DEVICE_CUDA) {
            CUDA_CHECK(cudaMemcpy(t->grad, init_grad, t->size * sizeof(float), cudaMemcpyHostToDevice));
        }
        else if(t->device == DEVICE_CPU) {
            memcpy((void*)t->grad, init_grad, t->size*sizeof(float));// init_grad;
        }
        free(init_grad);
    }

    
    if (t->n_parents!=0) {
        t->backward(t); // Calculate the gradients of the parents
        for (int i = 0; i < t->n_parents; i++) {
            // Tensor* p = t->parents[i];
            // Tensor* grad_tmp;            
            // CUDA_CHECK(cudaMalloc((void**) &grad_tmp, p->size * sizeof(float)));
            // CUDA_CHECK(cudaMemcpy(grad_tmp, p->grad, p->size*sizeof(float, cuda)))
            tensor_backward(t->parents[i], t->parents[i]->grad); // recursive backward all the way to the root of the graph
            
        }
    }
}


extern "C"
Tensor* tensor_to_cuda(const Tensor* src) {
    if (!src) return NULL;

    if (src->device == DEVICE_CUDA) {
        fprintf(stderr, "tensor_to_cuda: tensor already on CUDA\n");
        return (Tensor*)src;
    }


    Tensor* dst = (Tensor*)malloc(sizeof(Tensor));
    if (!dst) {
        fprintf(stderr, "tensor_to_cuda: failed to allocate Tensor\n");
        return NULL;
    }

    dst->ndim = src->ndim;
    dst->size = src->size;
    dst->requires_grad = src->requires_grad;
    dst->parents = src->parents;     // not copying graph here (for now)
    dst->n_parents = src->n_parents;
    dst->backward = NULL;
    dst->device = DEVICE_CUDA;

    // Copy shape and strides to host memory
    dst->shape = (int*)malloc(dst->ndim * sizeof(int));
    dst->strides = (int*)malloc(dst->ndim * sizeof(int));
    if (!dst->shape || !dst->strides) {
        fprintf(stderr, "tensor_to_cuda: failed to allocate shape/strides\n");
        free(dst->shape);
        free(dst->strides);
        free(dst);
        return NULL;
    }
    memcpy(dst->shape, src->shape, dst->ndim * sizeof(int));
    memcpy(dst->strides, src->strides, dst->ndim * sizeof(int));

    printf("inside tensor_to_cuda src:\n");
    print_tensor_info(src);
    Tensor* out = create_tensor(src->data, src->shape, src->ndim, src->requires_grad, DEVICE_CUDA);
    // Allocate device memory for data
    CUDA_CHECK(cudaMalloc((void**)&dst->data, dst->size * sizeof(float)));

    // Copy data from host (src->data) to device (dst->data)
    CUDA_CHECK(cudaMemcpy(dst->data,
                          src->data,
                          dst->size * sizeof(float),
                          cudaMemcpyHostToDevice));

    // // Not handling gradients on GPU yet
    // dst->grad = NULL;
    return out;
    // return dst;
}
extern "C"
Tensor* tensor_to_cpu(const Tensor* src) {
    // printf("p0- ");
    if (!src) return NULL;

    if (src->device == DEVICE_CPU) {
        fprintf(stderr, "tensor_from_cuda: tensor already on CPU\n");
        return NULL;
    }

    // printf("p1- ");
    Tensor* dst = create_empty_tensor(src->shape,
                                    src->ndim,
                                    src->requires_grad,
                                    DEVICE_CPU);
    

    // printf("p2- ");
    CUDA_CHECK(cudaMemcpy(dst->data, 
                        src->data,
                        src->size * sizeof(float),
                    cudaMemcpyDeviceToHost ));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    if(src->requires_grad==1) {
        CUDA_CHECK(cudaMemcpy(dst->grad, 
                            src->grad,
                            src->size * sizeof(float),
                        cudaMemcpyDeviceToHost ));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // printf("p3- ");
    if (!dst->shape || !dst->strides || !dst->data) {
        fprintf(stderr, "tensor_fom_cuda: failed to allocate shape/strides\n");
        free(dst->shape);
        free(dst->strides);
        free(dst->data);
        free(dst);
        return NULL;
    }
    // print_tensor_info(src);
    // printf("p4- %f", dst->data[0]);


    return dst;
}

