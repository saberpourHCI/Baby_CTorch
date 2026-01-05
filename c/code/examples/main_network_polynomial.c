
#include <math.h>
#include "../include/tensor.h"
#include "../include/tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>


#include "../include/model.h"
#include "../include/linear.h"
#include "../include/activation.h"
#include "../include/loss.h"
#include "../include/cuda_utils.h"
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
        x_data[i] = 2.0*(float)(i)/(float)(N) - 1.0;           // input
        // y_data[i] = (float)(3*i+1)/(float)(3*N+1);// (float)(i*i)/(float)(N*N);// sinf(t);//(N);     // target
        y_data[i] = (float)(i*i*i)/(float)(N*N*N);// sinf(t);//(N);     // target
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
    Tensor* h = linear_forward(l1, x);
    h = tanh_autograd(h);
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



//Read the parameters
// int epochs = 0;
// float lr = 0.0f;
// char key[64];
// char value[64];

// FILE *file = fopen("param.txt", "r");
// if (file == NULL) {
//     perror("Error opening file");
//     return 1;
// }

// while (fscanf(file, "%63[^=]=%63s", key, value) == 2) {
//     if (strcmp(key, "epochs") == 0) {
//         epochs = atoi(value);
//     } else if (strcmp(key, "lr") == 0) {
//         lr = atof(value);
//     }
// }
// fclose(file);

// printf("\np2\n");
float lr = 0.2;
int epochs = 2000;
FILE *fp = fopen("loss_data.csv", "w");
    if (!fp) {
        perror("Failed to open file");
        return 1;
    }

    fprintf(fp, "epoch,loss\n");  // CSV header


printf("entered for loop.\n");
int m = epochs - 1;
for (int epoch = 0; epoch < epochs; ++epoch) {
        // if (epoch<= epochs/2) {
        //     lr = 0.000001;
        // }
        // else {
        //     lr = 0.000001;
        // }
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
            printf("\nsum (output): *************************************** \n");
            t2 = tensor_to_cpu(t1->parents[0]);
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
            printf("\nlinear2 b: *************************************** \n");
            t2 = tensor_to_cpu(t1->parents[0]->parents[0]->parents[0]->parents[0]->parents[1]);
            print_tensor_info(t2);
            printf("\nlinear2 matmul (input): *************************************** \n");
            t2 = tensor_to_cpu(t1->parents[0]->parents[0]->parents[0]->parents[0]->parents[0]->parents[0]);
            print_tensor_info(t2);
            printf("\nlinear2 W: *************************************** \n");
            t2 = tensor_to_cpu(l2->W);
            print_tensor_info(t2);
            printf("\nrelu (input): *************************************** \n");
            t2 = tensor_to_cpu(t1->parents[0]->parents[0]->parents[0]->parents[0]->parents[0]->parents[0]->parents[0]);
            print_tensor_info(t2);

            printf("\nlinear1 b: *************************************** \n");
            t2 = tensor_to_cpu(t1->parents[0]->parents[0]->parents[0]->parents[0]->parents[0]->parents[0]->parents[0]->parents[1]);
            print_tensor_info(t2);
            printf("\nlinear1 matmul (input): *************************************** \n");
            t2 = tensor_to_cpu(t1->parents[0]->parents[0]->parents[0]->parents[0]->parents[0]->parents[0]->parents[0]->parents[0]->parents[0]);
            print_tensor_info(t2);
            printf("\nlinear1 W: *************************************** \n");
            t2 = tensor_to_cpu(l1->W);
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









/*
open "x64 Native Tools Command Prompt for VS22", run the following:

'''
cd D:\projects\ctorch\CPytorch\c
code .
'''

Then in the VS code that opens, open a "command prompt" terminal and run the following:

'''
nvcc code/src/tensor.cu code/src/cuda_utils.cu code/src/ops_add_sub_cpu.c code/src/ops_add_sub_cuda.cu code/src/ops_add_sub.c code/src/ops_mul_div_cpu.c code/src/ops_mul_div_cuda.cu code/src/ops_mul_div.c code/src/ops_matmul.c code/src/ops_matmul_cpu.c code/src/ops_matmul_cuda.cu code/src/linear.c code/src/activation_cpu.c code/src/activation_cuda.cu code/src/activation.c code/src/params.cu code/src/model.c code/src/loss.c code/examples/main_network_sin.c -o main1.exe
'''

This would generate the "main.exe" file, that you can execute later.






run the following in Git bash:

'''
chmod +x build.sh     # only once
./build.sh
./main.exe

'''
*/






