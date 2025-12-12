
#include <math.h>
#include "include/tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "include/model.h"
#include "include/linear.h"
#include "include/activation.h"
#include "include/loss.h"
#include "include/cuda_utils.h"
#include "cuda_runtime.h"



// void create_sine_dataset(int N,
//                          Tensor** x_cpu_out,
//                          Tensor** y_cpu_out)
                         
void create_sine_dataset(int N,
                         Tensor** x_cpu_out,
                         Tensor** y_cpu_out)
{
    float* x_data = (float*)malloc(N * sizeof(float));
    float* y_data = (float*)malloc(N * sizeof(float));
    if (!x_data || !y_data) {
        fprintf(stderr, "create_sine_dataset: malloc failed\n");
        return;
    }

    const float two_pi = 2.0f * 3.1415926535f;
    for (int i = 0; i < N; ++i) {
        float t = (two_pi * i) / (float)N;
        x_data[i] = i;           // input
        y_data[i] = (float)(3*i+1)/(float)(3*N+1);// (float)(i*i)/(float)(N*N);// sinf(t);//(N);     // target
        // y_data[i] = exp(-0.1 * t) * sin(2.0 * t);
    }

    int shape[2] = { N, 1 };

    Tensor* x_cpu = create_tensor(x_data, shape, 2, 1, DEVICE_CPU);
    Tensor* y_cpu = create_tensor(y_data, shape, 2, 1, DEVICE_CPU);
    print_tensor_info(x_cpu);

    x_cpu->requires_grad = 0;
    y_cpu->requires_grad = 0;

    // x_data/y_data are copied into the tensors (assuming create_tensor does that),
    // so we can free the raw arrays now:
    free(x_data);
    free(y_data);

    *x_cpu_out = x_cpu;
    *y_cpu_out = y_cpu;
}

Tensor* forward(Linear* l1, Linear* l2, Tensor* x) {
    // printf("here is the first linear input: \n");
    // print_tensor_info(x);
    Tensor* h = linear_forward(l1, x);
    // printf("here is the first linear output: \n");
    // print_tensor_info(h);
    // h = relu_autograd(h);
    Tensor* y_pred = linear_forward(l2, h);
    return y_pred;
}



int main() {


float* d = malloc(sizeof(float*));
Tensor* x_cpu = malloc(sizeof(Tensor*));
Tensor* y = malloc(sizeof(Tensor*));

// Tensor** x_cpu = &x; 
// Tensor** y_cpu = &y;
create_sine_dataset(50, &x_cpu, &y);
printf("after create_sine\n");
print_tensor_info(x_cpu);
Tensor* x = tensor_to_cuda(x_cpu);
print_tensor_info(x);

Tensor* y_true  = tensor_to_cuda(y);
// print_tensor_info(x);

// return 0;

free_tensor(x_cpu); free_tensor(y);


// We don't need x/y grads
x->requires_grad = 0;
y_true->requires_grad = 0;

x->parents = NULL;
x->n_parents = 0;
x->backward = NULL;
y_true->parents = NULL;
y_true->n_parents = 0;
y_true->backward = NULL;

Model* model = malloc(sizeof(Model*));
// printf("\np0\n");
model_init(model);
Linear* l1 = linear_create(model, 1, 16, DEVICE_CUDA);
Linear* l2 = linear_create(model, 16, 1, DEVICE_CUDA);


// printf("\np2\n");
float lr = 0.000001;
int epochs = 500;
FILE *fp = fopen("loss_data.csv", "w");
    if (!fp) {
        perror("Failed to open file");
        return 1;
    }

    fprintf(fp, "epoch,loss\n");  // CSV header


printf("entered for loop.\n");
int m = epochs - 1;
for (int epoch = 0; epoch < epochs; ++epoch) {
        if (epoch<= epochs/2) {
            lr = 0.000001;
        }
        else {
            lr = 0.000001;
        }
        // printf("\np3\n");
        // Forward
        printf("\n\nForward-------------------------------------------->\n\n");
        printf("\nforward: ");
        Tensor* y_pred = forward(l1, l2, x);      // on CUDA
        Tensor* loss   = MSE(y_pred, y_true);
        model_zero_grad(model);
        int shape[1] = {1};
        Tensor* root_grad = tensor_ones(shape, 1, 0, loss->device);
        
        tensor_backward(loss, root_grad->data);   // assume NULL means grad=1 for scalar
        printf("\n");
        model_sgd_step(model, lr);
        // printf("\np-1\n");

        // printf("\np6\n");
        printf("\n\nBackward-------------------------------------------->\n\n");
        // float* root_grad;
        // cudaMalloc((void**)&root_grad, sizeof(float));

        // printf("\np4\n");
        // printf("here is the y_pred: \n");
        // print_tensor_info(y_pred);
        Tensor* t1 = tensor_to_cpu(loss);
        // if(epoch == m) {
        //     printf("before step: ##########################################################################");
        //     // Tensor* t = tensor_to_cpu(l1->W);
        //     int cnt=0;
        //     // while(t1 != NULL) {
        //     //     printf("\n%d:", cnt++);
        //     printf("\nloss grad: *************************************** \n");
        //     Tensor* t2 = tensor_to_cpu(loss);
        //     print_tensor_info(t2);
        //     printf("\nsum (input): *************************************** \n");
        //     t2 = tensor_to_cpu(t1->parents[0]->parents[0]);
        //     print_tensor_info(t2);
        //     printf("\nsqr (input): *************************************** \n");
        //     t2 = tensor_to_cpu(t1->parents[0]->parents[0]->parents[0]);
        //     print_tensor_info(t2);
        //     printf("\ndiff (input): *************************************** \n");
        //     t2 = tensor_to_cpu(t1->parents[0]->parents[0]->parents[0]->parents[0]);
        //     print_tensor_info(t2);
        //     printf("\nlinear add (input): *************************************** \n");
        //     t2 = tensor_to_cpu(t1->parents[0]->parents[0]->parents[0]->parents[0]->parents[0]);
        //     print_tensor_info(t2);
        //     printf("\nlinear matmul (input): *************************************** \n");
        //     t2 = tensor_to_cpu(t1->parents[0]->parents[0]->parents[0]->parents[0]->parents[0]->parents[1]);
        //     print_tensor_info(t2);
        //     printf("\nlinear matmul (input): *************************************** \n");
        //     t2 = tensor_to_cpu(l2->W);
        //     print_tensor_info(t2);

        // }
        
       
        
        // printf("\np7\n");
        // Optimizer step
        // printf("\np0\n");

             // scalar tensor on CUDA
        // printf("\nhere is the loss: \n");
        // print_tensor_info(loss);
        // Tensor* tmp = tensor_to_cpu(loss);
        // printf("\nloss value is %f\n", tmp->data[0]);
        // printf("\np5\n");
        // Backward
        // printf("\np-2\n");
        

        t1 = loss;
        if(epoch == m) {
            printf("After step: ##########################################################################");
            // Tensor* t = tensor_to_cpu(l1->W);
            int cnt=0;
            // while(t1 != NULL) {
            //     printf("\n%d:", cnt++);
            printf("\nloss grad: *************************************** \n");
            Tensor* t2 = tensor_to_cpu(t1);
            print_tensor_info(t2);
            printf("\nsum (input): *************************************** \n");
            t2 = tensor_to_cpu(t1->parents[0]->parents[0]);
            print_tensor_info(t2);
            printf("\nsqr (input): *************************************** \n");
            t2 = tensor_to_cpu(t1->parents[0]->parents[0]->parents[0]);
            print_tensor_info(t2);
            printf("\ndiff (input): *************************************** \n");
            t2 = tensor_to_cpu(t1->parents[0]->parents[0]->parents[0]->parents[0]);
            print_tensor_info(t2);
            printf("\nlinear b: *************************************** \n");
            t2 = tensor_to_cpu(t1->parents[0]->parents[0]->parents[0]->parents[0]->parents[1]);
            print_tensor_info(t2);
            printf("\nlinear matmul (input): *************************************** \n");
            t2 = tensor_to_cpu(t1->parents[0]->parents[0]->parents[0]->parents[0]->parents[0]->parents[0]);
            print_tensor_info(t2);
            printf("\nlinear W: *************************************** \n");
            t2 = tensor_to_cpu(l2->W);
            print_tensor_info(t2);
            printf("\nrelu (input): *************************************** \n");
            t2 = tensor_to_cpu(t1->parents[0]->parents[0]->parents[0]->parents[0]->parents[0]->parents[0]->parents[0]);
            print_tensor_info(t2);

            

            // t1 = t1->parents[0];
            // }
            // break;
        }
        // Tensor* l = tensor_from_cuda(loss);
        
        // printf("\np1\n");
        // printf("\nepoch: %d\n loss: ", epoch, l->data[0]);
        printf("\nepoch: %d\n", epoch);
        // free(l);
        print_tensor_info(loss);
        printf("\nss_cpu is: ");
        Tensor* loss_cpu = tensor_to_cpu(loss);
        printf("-->%f\n", loss_cpu->data[0]);
        fprintf(fp, "%d,%f\n", epoch, loss_cpu->data[0]);

        free_tensor(y_pred);
        free_tensor(loss);
    }
    // Tensor* y_pred = linear_forward(l1, x);
    // y_pred = relu_autograd(y_pred);
    // y_pred = linear_forward(l2, y_pred);
    // y_pred = l1->W;
    Tensor* y_true_cpu = tensor_to_cpu(y_true);
    printf("\n Here is the y_true : \n");
    print_tensor_info(y_true_cpu);
    
    Tensor* y_pred = forward(l1, l2, x);
    Tensor* y_cpu = tensor_to_cpu(y_pred);
    printf("\n Here is the y_cpu : \n");
    print_tensor_info(y_cpu);
    // x_cpu = tensor_to_cpu(x);
    // for(int i=0; i<x->size; i++) {
    //     fprintf(fp, "%d,%f\n", i, y_cpu->data[i]);
    // }
    fclose(fp);






    // Call Python to plot
    int ret = system("python ../py/plot.py");
    if (ret != 0) {
        fprintf(stderr, "Failed to run Python script\n");
    }










    // Cleanup
    free_tensor(x);
    free_tensor(y_true);
    linear_free(l1);
    linear_free(l2);
    model_free(model);





return 0;
}


















// #include "tensor.h"
// #include "ops_add_sub.h"
// #include "ops_mul_div.h"
// #include "ops_matmul.h"
// #include "loss.h"
// #include "activation.h"
// #include "linear.h"
// #include <stdio.h>
// #include <stdlib.h>
// #include <string.h>
// #include <stdbool.h>



// /*typedef struct Tensor Tensor;*/
// typedef void (*BackwardFn)(Tensor*);

// /*struct Tensor {
//     float* data;
//     float* grad;
//     int* shape;
//     int* strides;
//     int ndim;
//     int size;

//     // Autograd parameters
//     int requires_grad;    // 1 = track gradients
//     Tensor** parents;     // Array of parent tensors
//     int n_parents;        // Number of parent tensors
//     BackwardFn backward;  // Function for gradients computation
// };*/



// /*void free_tensor(Tensor* tensor) {
//     if (!tensor) return;
//     free(tensor->shape);
//     free(tensor->strides);
//     free(tensor);
// }*/


// /*Tensor* create_tensor(float* data, const int* shape, int ndim) {
//     if (data == NULL || shape == NULL || ndim <= 0) {
//         fprintf(stderr, "Invalid input parameters\n");
//         return NULL;
//     }

//     Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
//     if (tensor == NULL) {
//         fprintf(stderr, "Failed allocating memory for tensor\n");
//         return NULL;
//     }

//     tensor->data = data;
//     tensor->shape = (int*)malloc(ndim * sizeof(int));
//     if (tensor->shape == NULL) {
//         fprintf(stderr, "Memory allocation failed\n");
//         free(tensor);
//         return NULL;
//     }
//     memcpy(tensor->shape, shape, ndim * sizeof(int));
//     tensor->ndim = ndim;

//     tensor->size = 1;
//     for (int i = 0; i < ndim; i++) {
//         tensor->size *= shape[i];
//     }

//     tensor->strides = (int*)malloc(ndim * sizeof(int));
//     if (tensor->strides == NULL) {
//         fprintf(stderr, "Memory allocation failed\n");
//         free(tensor->shape);
//         free(tensor);
//         return NULL;
//     }
//     int stride = 1;
//     for (int i = ndim - 1; i >= 0; i--) {
//         tensor->strides[i] = stride;
//         stride *= shape[i];
//     }

//     return tensor;
// }*/


// /*Tensor* create_tensor_autograd(float* data, const int* shape, int ndim, int requires_grad) {
//     Tensor* t = create_tensor(data, shape, ndim);
//     if (!t) return NULL;

//     t->requires_grad = requires_grad;
//     t->grad = NULL;
//     t->parents = NULL;
//     t->n_parents = 0;
//     t->backward = NULL;

//     if (requires_grad) {
//         t->grad = (float*)calloc(t->size, sizeof(float));
//         if (!t->grad) {
//             free_tensor(t);
//             return NULL;
//         }
//     }

//     return t;
// }*/


// /*void backward_add(Tensor* out) {
//     Tensor* A = out->parents[0];
//     Tensor* B = out->parents[1];

//     for (int i = 0; i < A->size; i++)
//         A->grad[i] += out->grad[i];

//     for (int i = 0; i < B->size; i++)
//         B->grad[i] += out->grad[i];
// }*/


// /*void backward_sub(Tensor* out) {
//     Tensor* A = out->parents[0];
//     Tensor* B = out->parents[1];

//     for (int i = 0; i < A->size; i++)
//         A->grad[i] += out->grad[i];

//     for (int i = 0; i < B->size; i++)
//         B->grad[i] -= out->grad[i];
// }*/


// /*void backward_mul(Tensor* out) {

//     Tensor* A = out->parents[0];
//     Tensor* B = out->parents[1];

//     for (int i = 0; i < out->size; i++) {
//         int idx_a = 0, idx_b = 0, rem = i;
//         for (int d = out->ndim - 1; d >= 0; d--) {
//             int coord = rem % out->shape[d];
//             rem /= out->shape[d];

//             int a_has = (d >= out->ndim - A->ndim);
//             int b_has = (d >= out->ndim - B->ndim);

//             int a_dim = a_has ? A->shape[d - (out->ndim - A->ndim)] : 1;
//             int b_dim = b_has ? B->shape[d - (out->ndim - B->ndim)] : 1;

//             int a_str = a_has ? A->strides[d - (out->ndim - A->ndim)] : 0;
//             int b_str = b_has ? B->strides[d - (out->ndim - B->ndim)] : 0;

//             if (a_dim != 1) idx_a += coord * a_str;

//             if (b_dim != 1) idx_b += coord * b_str;
//         }

//         float g = out->grad ? out->grad[i] : 1.0f;

//         // Chain rule:
//         //   dL/dA[idx_a] += dL/dout[i] * B[idx_b]
//         //   dL/dB[idx_b] += dL/dout[i] * A[idx_a]
//         // Note: multiple i can map to the same idx_a/idx_b (broadcast reduction),
//         // so we accumulate (+=) instead of assigning.
//         if (A->grad) A->grad[idx_a] += B->data[idx_b] * g;
//         if (B->grad) B->grad[idx_b] += A->data[idx_a] * g;
//     }
// }*/


// /*void backward_div(Tensor* out) {

//     Tensor* A = out->parents[0];
//     Tensor* B = out->parents[1];

//     for (int i = 0; i < out->size; i++) {
//         int idx_a = 0, idx_b = 0, rem = i;
//         for (int d = out->ndim - 1; d >= 0; d--) {
//             int coord = rem % out->shape[d];
//             rem /= out->shape[d];

//             int a_has = (d >= out->ndim - A->ndim);
//             int b_has = (d >= out->ndim - B->ndim);

//             int a_dim = a_has ? A->shape[d - (out->ndim - A->ndim)] : 1;
//             int b_dim = b_has ? B->shape[d - (out->ndim - B->ndim)] : 1;

//             int a_str = a_has ? A->strides[d - (out->ndim - A->ndim)] : 0;
//             int b_str = b_has ? B->strides[d - (out->ndim - B->ndim)] : 0;

//             if (a_dim != 1) idx_a += coord * a_str;

//             if (b_dim != 1) idx_b += coord * b_str;
//         }

//         float g = out->grad ? out->grad[i] : 1.0f;
//         float b = B->data[idx_b];

//         printf("%f - ",b);
//         printf("\n");
//         if (A->grad) A->grad[idx_a] += 1.0/b * g;
//         if (B->grad) B->grad[idx_b] += -(A->data[idx_a]/(b* b)) * g;
//         printf("A->grad[%d] is %f \n", idx_a, A->grad[idx_a]);
//     }
// }*/


// /*void print_tensor_info(const Tensor* t) {
//     printf("Tensor: ndim=%d, size=%d\n", t->ndim, t->size);
//     printf("Shape: [");
//     for (int i = 0; i < t->ndim; i++) {
//         printf("%d%s", t->shape[i], i == t->ndim - 1 ? "" : ", ");
//     }
//     printf("]\nStrides: [");
//     for (int i = 0; i < t->ndim; i++) {
//         printf("%d%s", t->strides[i], i == t->ndim - 1 ? "" : ", ");
//     }
//     printf("]\n");
// }*/


// /*int compute_size(const int* shape, int ndim) {
//     int size = 1;
//     for (int i = 0; i < ndim; i++)
//         size *= shape[i];
//     return size;
// }*/


// /*int* broadcast_shapes(const int* a_shape, int a_ndim, const int* b_shape, int b_ndim, int* out_ndim) {
//     int ndim = (a_ndim > b_ndim) ? a_ndim : b_ndim;
//     int* result_shape = (int*)malloc(ndim * sizeof(int));
//     if (!result_shape) return NULL;

//     for (int i = 0; i < ndim; i++) {
//         int a_dim = (i >= ndim - a_ndim) ? a_shape[i - (ndim - a_ndim)] : 1;
//         int b_dim = (i >= ndim - b_ndim) ? b_shape[i - (ndim - b_ndim)] : 1;

//         if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
//             free(result_shape);
//             return NULL; // Incompatible shapes
//         }
//         result_shape[i] = (a_dim > b_dim) ? a_dim : b_dim;
//     }

//     *out_ndim = ndim;
//     return result_shape;
// }*/



// /*Tensor* tensor_add(const Tensor* a, const Tensor* b) {
//     int out_ndim;
//     int* out_shape = broadcast_shapes(a->shape, a->ndim, b->shape, b->ndim, &out_ndim);
//     if (!out_shape) {
//         fprintf(stderr, "Error: Incompatible shapes for addition\n");
//         return NULL;
//     }

//     int out_size = compute_size(out_shape, out_ndim);
//     float* out_data = (float*)malloc(out_size * sizeof(float));
//     if (!out_data) {
//         free(out_shape);
//         fprintf(stderr, "Error: Memory allocation failed\n");
//         return NULL;
//     }

//     // Iterate through all elements using linear indexing
//     for (int i = 0; i < out_size; i++) {
//         // Compute multi-dimensional index
//         int idx_a = 0, idx_b = 0;
//         int rem = i;
//         for (int d = out_ndim - 1; d >= 0; d--) {
//             int coord = rem % out_shape[d];
//             rem /= out_shape[d];

//             int a_dim = (d >= out_ndim - a->ndim) ? a->shape[d - (out_ndim - a->ndim)] : 1;
//             int b_dim = (d >= out_ndim - b->ndim) ? b->shape[d - (out_ndim - b->ndim)] : 1;

//             int a_stride = (d >= out_ndim - a->ndim) ? a->strides[d - (out_ndim - a->ndim)] : 0;
//             int b_stride = (d >= out_ndim - b->ndim) ? b->strides[d - (out_ndim - b->ndim)] : 0;

//             if (a_dim != 1) idx_a += coord * a_stride;
//             if (b_dim != 1) idx_b += coord * b_stride;
//         }

//         out_data[i] = a->data[idx_a] + b->data[idx_b];
//     }

//     Tensor* out = create_tensor(out_data, out_shape, out_ndim);
//     free(out_shape);
//     return out;
// }*/

// /*Tensor* tensor_sub(const Tensor* a, const Tensor* b) {
//     int out_ndim;
//     int* out_shape = broadcast_shapes(a->shape, a->ndim, b->shape, b->ndim, &out_ndim);
//     if (!out_shape) {
//         fprintf(stderr, "Error: Incompatible shapes for addition\n");
//         return NULL;
//     }

//     int out_size = compute_size(out_shape, out_ndim);
//     float* out_data = (float*)malloc(out_size * sizeof(float));
//     if (!out_data) {
//         free(out_shape);
//         fprintf(stderr, "Error: Memory allocation failed\n");
//         return NULL;
//     }

//     // Iterate through all elements using linear indexing
//     for (int i = 0; i < out_size; i++) {
//         // Compute multi-dimensional index
//         int idx_a = 0, idx_b = 0;
//         int rem = i;
//         for (int d = out_ndim - 1; d >= 0; d--) {
//             int coord = rem % out_shape[d];
//             rem /= out_shape[d];

//             int a_dim = (d >= out_ndim - a->ndim) ? a->shape[d - (out_ndim - a->ndim)] : 1;
//             int b_dim = (d >= out_ndim - b->ndim) ? b->shape[d - (out_ndim - b->ndim)] : 1;

//             int a_stride = (d >= out_ndim - a->ndim) ? a->strides[d - (out_ndim - a->ndim)] : 0;
//             int b_stride = (d >= out_ndim - b->ndim) ? b->strides[d - (out_ndim - b->ndim)] : 0;

//             if (a_dim != 1) idx_a += coord * a_stride;
//             if (b_dim != 1) idx_b += coord * b_stride;
//         }

//         out_data[i] = a->data[idx_a] - b->data[idx_b];
//     }

//     Tensor* out = create_tensor(out_data, out_shape, out_ndim);
//     free(out_shape);
//     return out;
// }*/

// /*Tensor* tensor_mul(const Tensor* a, const Tensor* b) {
//     int out_ndim;
//     int* out_shape = broadcast_shapes(a->shape, a->ndim, b->shape, b->ndim, &out_ndim);
//     if (!out_shape) {
//         fprintf(stderr, "Error: Incompatible shapes for addition\n");
//         return NULL;
//     }

//     int out_size = compute_size(out_shape, out_ndim);
//     float* out_data = (float*)malloc(out_size * sizeof(float));
//     if (!out_data) {
//         free(out_shape);
//         fprintf(stderr, "Error: Memory allocation failed\n");
//         return NULL;
//     }

//     // Iterate through all elements using linear indexing
//     for (int i = 0; i < out_size; i++) {
//         // Compute multi-dimensional index
//         int idx_a = 0, idx_b = 0;
//         int rem = i;
//         for (int d = out_ndim - 1; d >= 0; d--) {
//             int coord = rem % out_shape[d];
//             rem /= out_shape[d];

//             int a_dim = (d >= out_ndim - a->ndim) ? a->shape[d - (out_ndim - a->ndim)] : 1;
//             int b_dim = (d >= out_ndim - b->ndim) ? b->shape[d - (out_ndim - b->ndim)] : 1;

//             int a_stride = (d >= out_ndim - a->ndim) ? a->strides[d - (out_ndim - a->ndim)] : 0;
//             int b_stride = (d >= out_ndim - b->ndim) ? b->strides[d - (out_ndim - b->ndim)] : 0;

//             if (a_dim != 1) idx_a += coord * a_stride;
//             if (b_dim != 1) idx_b += coord * b_stride;
//         }

//         out_data[i] = a->data[idx_a] * b->data[idx_b];
//     }

//     Tensor* out = create_tensor(out_data, out_shape, out_ndim);
//     free(out_shape);
//     return out;
// }*/

// /*Tensor* tensor_div(const Tensor* a, const Tensor* b) {
//     int out_ndim;
//     int* out_shape = broadcast_shapes(a->shape, a->ndim, b->shape, b->ndim, &out_ndim);
//     if (!out_shape) {
//         fprintf(stderr, "Error: Incompatible shapes for addition\n");
//         return NULL;
//     }

//     int out_size = compute_size(out_shape, out_ndim);
//     float* out_data = (float*)malloc(out_size * sizeof(float));
//     if (!out_data) {
//         free(out_shape);
//         fprintf(stderr, "Error: Memory allocation failed\n");
//         return NULL;
//     }

//     // Iterate through all elements using linear indexing
//     for (int i = 0; i < out_size; i++) {
//         // Compute multi-dimensional index
//         int idx_a = 0, idx_b = 0;
//         int rem = i;
//         for (int d = out_ndim - 1; d >= 0; d--) {
//             int coord = rem % out_shape[d];
//             rem /= out_shape[d];

//             int a_dim = (d >= out_ndim - a->ndim) ? a->shape[d - (out_ndim - a->ndim)] : 1;
//             int b_dim = (d >= out_ndim - b->ndim) ? b->shape[d - (out_ndim - b->ndim)] : 1;

//             int a_stride = (d >= out_ndim - a->ndim) ? a->strides[d - (out_ndim - a->ndim)] : 0;
//             int b_stride = (d >= out_ndim - b->ndim) ? b->strides[d - (out_ndim - b->ndim)] : 0;

//             if (a_dim != 1) idx_a += coord * a_stride;
//             if (b_dim != 1) idx_b += coord * b_stride;
//         }

//         float denom = b->data[idx_b];
//         // simple guard; you can choose to error out instead
//         if (denom == 0.0f) {
//             fprintf(stderr, "Warning: division by zero at element %d, setting to 0\n", i);
//             out_data[i] = 0.0f;
//         } else {
//             out_data[i] = a->data[idx_a] / denom;
//         }
//     }
//     int out_requires_grad = (a->requires_grad==1 && b->requires_grad==1)?1:0;
//     // printf("out_requries_grad is ====> %d\n", out_requires_grad);
//     Tensor* out = create_tensor_autograd(out_data, out_shape, out_ndim, out_requires_grad);
//     free(out_shape);
//     return out;
// }*/



// /*Tensor* tensor_add_autograd(Tensor* A, Tensor* B) {
//     Tensor* out = tensor_add(A, B);
//     if (!out) return NULL;

//     if (A->requires_grad || B->requires_grad) {
//         out->requires_grad = 1;
//         out->parents = (Tensor**)malloc(2 * sizeof(Tensor*));
//         out->parents[0] = A;
//         out->parents[1] = B;
//         out->n_parents = 2;
//         out->backward = backward_add;
//         out->grad = (float*)calloc(out->size, sizeof(float));
//     }

//     return out;
// }*/

// /*Tensor* tensor_sub_autograd(Tensor* A, Tensor* B) {
//     Tensor* out = tensor_sub(A, B);
//     if (!out) return NULL;

//     if (A->requires_grad || B->requires_grad) {
//         out->requires_grad = 1;
//         out->parents = (Tensor**)malloc(2 * sizeof(Tensor*));
//         out->parents[0] = A;
//         out->parents[1] = B;
//         out->n_parents = 2;
//         out->backward = backward_sub;
//         out->grad = (float*)calloc(out->size, sizeof(float));
//     }

//     return out;
// }*/

// /*Tensor* tensor_mul_autograd(Tensor* A, Tensor* B){
//     Tensor* out = tensor_mul(A, B);
//     if (!out) return NULL;

//     if (A->requires_grad || B->requires_grad) {
//         out->requires_grad = 1;
//         out->parents = (Tensor**)malloc(2 * sizeof(Tensor*));
//         out->parents[0] = A;
//         out->parents[1] = B;
//         out->n_parents = 2;
//         out->grad = (float*)calloc(out->size, sizeof(float));
//         out->backward = backward_mul;
        
//     }

//     return out;
// }*/

// /*Tensor* tensor_div_autograd(Tensor* A, Tensor* B){
//     Tensor* out = tensor_div(A, B);
//     if (!out) return NULL;

//     if (A->requires_grad || B->requires_grad) {
//         out->requires_grad = 1;
//         out->parents = (Tensor**)malloc(2 * sizeof(Tensor*));
//         out->parents[0] = A;
//         out->parents[1] = B;
//         out->n_parents = 2;
//         out->grad = (float*)calloc(out->size, sizeof(float));
//         out->backward = backward_div;
//         printf("p1: ----> B->grad[0] = %f\n", B->grad[0]);
        
        
//     }

//     return out;
// }*/



// /*// Uses broadcasting for the last 2 dimensions and accounts for the batch multiplicartion
// Tensor* tensor_matmul(const Tensor* A, const Tensor* B) {
//     if (A->ndim < 2 || B->ndim < 2) {
//         fprintf(stderr, "Error: Matmul requires ndim >= 2\n");
//         return NULL;
//     }

//     int A_m = A->shape[A->ndim - 2];
//     int A_k = A->shape[A->ndim - 1];
//     int B_k = B->shape[B->ndim - 2];
//     int B_n = B->shape[B->ndim - 1];

//     if (A_k != B_k) {
//         fprintf(stderr, "Error: Inner dimensions must match for matmul\n");
//         return NULL;
//     }

//     //Computing broadcasted batch shape===========================================
//     int A_batch_ndim = A->ndim - 2;
//     int B_batch_ndim = B->ndim - 2;
//     int out_batch_ndim = (A_batch_ndim > B_batch_ndim) ? A_batch_ndim : B_batch_ndim;

//     int* out_batch_shape = (int*)malloc(out_batch_ndim * sizeof(int));
//     if (!out_batch_shape) {
//         printf("b0");
//         return NULL;
//     }

//     for (int i = 0; i < out_batch_ndim; i++) {
//         int a_dim = (i >= out_batch_ndim - A_batch_ndim) ? A->shape[i - (out_batch_ndim - A_batch_ndim)] : 1;
//         int b_dim = (i >= out_batch_ndim - B_batch_ndim) ? B->shape[i - (out_batch_ndim - B_batch_ndim)] : 1;

//         if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
//             fprintf(stderr, "Error: Incompatible batch dimensions for matmul\n");
//             free(out_batch_shape);
//             return NULL;
//         }
//         out_batch_shape[i] = (a_dim > b_dim) ? a_dim : b_dim;
//     }

//     //Create final output shape ===========================================
//     int out_ndim = out_batch_ndim + 2;
//     int* out_shape = (int*)malloc(out_ndim * sizeof(int));
//     if (!out_shape) {
//         printf("b1");
//         free(out_batch_shape);
//         return NULL;
//     }

//     memcpy(out_shape, out_batch_shape, out_batch_ndim * sizeof(int));
//     out_shape[out_ndim - 2] = A_m;
//     out_shape[out_ndim - 1] = B_n;

//     int out_size = compute_size(out_shape, out_ndim);
//     float* out_data = (float*)calloc(out_size, sizeof(float));
//     if (!out_data) {
//         free(out_batch_shape);
//         free(out_shape);
//         fprintf(stderr, "Error: Memory allocation failed\n");
//         return NULL;
//     }

//     //Iterate over all broadcasted batches ===========================================
//     int total_batches = compute_size(out_batch_shape, out_batch_ndim);

//     for (int batch_idx = 0; batch_idx < total_batches; batch_idx++) {
//         // Compute multi-dimensional batch index
//         int rem = batch_idx;
//         int a_offset = 0;
//         int b_offset = 0;

//         for (int d = out_batch_ndim - 1; d >= 0; d--) {
//             int coord = rem % out_batch_shape[d];
//             rem /= out_batch_shape[d];

//             int a_dim = (d >= out_batch_ndim - A_batch_ndim) ? A->shape[d - (out_batch_ndim - A_batch_ndim)] : 1;
//             int b_dim = (d >= out_batch_ndim - B_batch_ndim) ? B->shape[d - (out_batch_ndim - B_batch_ndim)] : 1;

//             int a_stride = (d >= out_batch_ndim - A_batch_ndim) ? A->strides[d - (out_batch_ndim - A_batch_ndim)] : 0;
//             int b_stride = (d >= out_batch_ndim - B_batch_ndim) ? B->strides[d - (out_batch_ndim - B_batch_ndim)] : 0;

//             if (a_dim != 1) a_offset += coord * a_stride;
//             if (b_dim != 1) b_offset += coord * b_stride;
//         }

//         // a_offset *= (A_m * A_k);
//         // b_offset *= (B_k * B_n);
//         int c_offset = batch_idx * (A_m * B_n);

//         //Matrix multiplication per batch ===========================================
//         for (int i = 0; i < A_m; i++) {
//             for (int j = 0; j < B_n; j++) {
//                 float sum = 0.0f;
//                 for (int k = 0; k < A_k; k++) {
//                     float a_val = A->data[a_offset + i * A_k + k];
//                     float b_val = B->data[b_offset + k * B_n + j];
//                     sum += a_val * b_val;
//                 }
//                 out_data[c_offset + i * B_n + j] = sum;
//             }
//         }
//     }

//     Tensor* out = create_tensor(out_data, out_shape, out_ndim);
//     free(out_shape);
//     free(out_batch_shape);
//     return out;
// }*/


// /*void tensor_backward(Tensor* t, float* grad) {
//     if (!t->grad) {
//         t->grad = (float*)calloc(t->size, sizeof(float));
//         for(int i =0; i < t->size; i++)
//             t->grad[i] = 10.0f;
//     }
 
//     if (grad != NULL) {
//         printf("NOT-null================> t->grad is: %f, and grad is: %f\n", t->grad[0], grad[0]);
//     } else {
//         printf("null================> t->grad is: %f, and grad is: %f\n", t->grad[0], grad);
//         for (int i = 0; i < t->size; i++)
//             t->grad[i] = 1.0f;
//     }

//     if (t->backward) {
//         t->backward(t); // Calculate the gradients of the parents
//         for (int i = 0; i < t->n_parents; i++) {
//             printf("t->grad is: %f ####################################################\n", t->grad[0]);
//             tensor_backward(t->parents[i], t->grad); // recursive backward all the way to the root of the graph
//         }
//     }
// }*/



// int main() {
//     float data1[3] = {1,2,3};
//     int shape1[1] = {3};
//     int ndim1 = 1;



//     float data2[6] = {4,5,6,7,8,9};
//     int shape2[2] = {2,3};
//     int ndim2 = 2;

//     float data3[6] = {4,5,6,7,8,9};
//     int shape3[2] = {3,2};
//     int ndim3 = 2;

//     float data4[1] = {2};
//     int shape4[1] = {1};
//     int ndim4 = 1;

//     Tensor* a = create_tensor_autograd(data1, shape1, ndim1, 1, DEVICE_CPU);
//     Tensor* b = create_tensor_autograd(data2, shape2, ndim2, 1, DEVICE_CPU);
//     Tensor* a0 = create_tensor_autograd(data4, shape4, ndim4, 1, DEVICE_CPU);
    
//     Tensor* a1 = tensor_div_autograd(a, a0);
//     print_tensor_info(a1);

//     return 0;
//     printf("Here is the tensor info for A printed: \n");
//     // a = tensor_to_cuda(a);
//     print_tensor_info(a);
//     printf("p0- ");
//     Tensor* c = tensor_div_autograd(a, b);  // c = a * b
//     printf("p1- ");
//     Tensor* d = tensor_div_autograd(c, b);
//     printf("p2- ");
//     Tensor* j = tensor_add_autograd(a, b);
    

//     printf("d: ");
//     for (int i = 0; i < d->size; i++) printf("%f ", d->data[i]);
//     printf("\n");

//     // Tensor* e = tensor_to_cuda(d);
//     // Tensor* g = tensor_to_cuda(c);
//     // Tensor* h = tensor_sub(e,g);

//     Tensor* e = create_tensor_autograd(data1, shape1, ndim1, 1, DEVICE_CUDA);
//     Tensor* g = create_tensor_autograd(data2, shape2, ndim2, 1, DEVICE_CUDA);
//     Tensor* g_prime = create_tensor_autograd(data3, shape3, ndim3, 1, DEVICE_CUDA);
//     Tensor* h = tensor_mul_autograd(e,g);
//     Tensor* k = tensor_div_autograd(e,g);
//     printf("\nbefore tensor_backward is called!\n");
//     Tensor* s = tensor_matmul_autograd(g, g_prime);
//     // tensor_backward(s, NULL);
//     print_tensor_info(g);
//     Tensor* sum = tensor_sum_autograd(e);
//     tensor_backward(sum, NULL);
//     printf("\nhere is the sum info : \n");
//     print_tensor_info(e);
//     // printf("-> %f", sum->data[0]);
//     return 0;


//     tensor_backward(k,NULL);
//     tensor_backward(h,NULL);
    
//     printf("\nk ----------------------");
//     print_tensor_info(k);
//     printf("h ----------------------");
//     print_tensor_info(h);
//     printf("g ----------------------");
//     print_tensor_info(g);
//     printf("e ----------------------");
//     print_tensor_info(e);


//     printf("h->grad: \n");
//     for(int i=0; i<h->size; i++)
//         printf("%f ", h->grad[i]);
//     printf("==============================\n\n");
//     printf("e->grad: \n");
//     for(int i=0; i<e->size; i++)
//         printf("%f ", e->grad[i]);
//     printf("==============================\n\n");
//     printf("g->grad: \n");
//     for(int i=0; i<g->size; i++)
//         printf("%f ", g->grad[i]);
//     printf("==============================\n\n");

    
    
    

//     printf("=========================\n");
//     printf("tensor e info is: ");
//     print_tensor_info(e);
//     printf("tensor g info is: ");
//     print_tensor_info(g);
//     printf("tensor h info is: ");
//     print_tensor_info(h);
//     printf("=========================");

//     Tensor* f = tensor_from_cuda(e);
//     printf("\n\ntensor e->data is: %f \n\n", f->data[0]);

//     printf("tensor e device is: ");
//     print_tensor_info(e);

//     printf("\n\n\n\ntensor f device is: ");
//     print_tensor_info(f);

//     free_tensor(a);
//     free_tensor(b);
//     free_tensor(c);
//     free_tensor(d);
//     return 0;
// }




// /*
// int main() {
//     printf("artin \n");
//     // A: (2, 1, 2, 3)
//     // B: (1, 4, 3, 2)
//     // Expected output: (2, 4, 2, 2)
//     float A_data[2 * 1 * 2 * 3];
//     float B_data[1 * 4 * 3 * 2];

//     for (int i = 0; i < 12; i++) A_data[i] = i + 1;     // Fill with 1..12
//     for (int i = 0; i < 24; i++) B_data[i] = (i + 1) * 0.5;

//     int A_shape[4] = {2, 1, 2, 3};
//     int B_shape[4] = {1, 4, 3, 2};

//     Tensor* A = create_tensor(A_data, A_shape, 4);
//     Tensor* B = create_tensor(B_data, B_shape, 4);
//     printf("p0");
//     Tensor* C = tensor_matmul(A, B);
//     printf("p1");
//     if (!C) {
//         printf("C not initialized!");
//         return 1;
//     }

//     printf("Output shape: (");
//     for (int i = 0; i < C->ndim; i++)
//         printf("%d%s", C->shape[i], (i == C->ndim - 1) ? ")\n" : ", ");
//     printf("Total elements: %d\n", C->size);

//     // Print first few elements
//     for (int i = 0; i < (C->size < 10 ? C->size : 10); i++)
//         printf("%.2f ", C->data[i]);
//     printf("...\n");

//     free_tensor(A);
//     free_tensor(B);
//     free(C->data);
//     free_tensor(C);
//     return 0;
// }
// */



// /*int main() {
//     float data1[6] = {1, 2, 3, 4, 5, 6};
//     float data2[3] = {10, 20, 30};

//     int shape1[2] = {2, 3};
//     int shape2[1] = {3};

//     Tensor* A = create_tensor(data1, shape1, 2);
//     Tensor* B = create_tensor(data2, shape2, 1);

//     Tensor* C = tensor_mul(A, B);
//     if (!C) return 1;

//     printf("Result:\n");
//     for (int i = 0; i < C->size; i++) {
//         printf("%.2f ", C->data[i]);
//     }
//     printf("\n");

//     Tensor* D = tensor_sub(A, B);
//     if (!D) return 1;

//     printf("Result:\n");
//     for (int i = 0; i < D->size; i++) {
//         printf("%.2f ", D->data[i]);
//     }
//     printf("\n");

//     free_tensor(A);
//     free_tensor(B);
//     free(C->data); // Since tensor_add allocates data
//     free_tensor(C);
//     free(D->data); // Since tensor_add allocates data
//     free_tensor(D);
//     return 0;
// }
// */

// /*
// int main() {
//     float data1[3] = {1,2,3};
//     int shape1[1] = {3};
//     int ndim1 = 1;



//     float data2[6] = {4,5,6,7,8,9};
//     int shape2[2] = {2,3};
//     int ndim2 = 2;


//     Tensor* a = create_tensor_autograd(data1, shape1, ndim1, 1);
//     Tensor* b = create_tensor_autograd(data2, shape2, ndim2, 1);

//     Tensor* c = tensor_div_autograd(a, b);  // c = a * b

//     Tensor* d = tensor_div_autograd(c, b);
    
//     tensor_backward(d, NULL); // compute gradients

//     printf("grad a: ");
//     for (int i = 0; i < a->size; i++) printf("%f ", a->grad[i]);
//     printf("\n");

//     printf("grad b: ");
//     for (int i = 0; i < b->size; i++) printf("%f ", b->grad[i]);
//     printf("\n");

//     printf("grad c: ");
//     for (int i = 0; i < c->size; i++) printf("%f ", c->grad[i]);
//     printf("\n");


//     printf("grad d: ");
//     for (int i = 0; i < d->size; i++) printf("%f ", d->grad[i]);
//     printf("\n");

//     printf("d: ");
//     for (int i = 0; i < d->size; i++) printf("%f ", d->data[i]);
//     printf("\n");

//     free_tensor(a);
//     free_tensor(b);
//     free_tensor(c);
//     free_tensor(d);
//     return 0;
// }

// */