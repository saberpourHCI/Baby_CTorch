
#include <math.h>
#include "include/tensor.h"
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
#include "utils.c"



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
        y_data[i] = sinf(t);//(N);     // target
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

Tensor* forward(Linear** l, int num_l, Tensor* x) {
    Tensor* h = x;
    for(int i=0; i<num_l; i++) {
        h = linear_forward(l[i], h);
        if(i!=num_l-1) {
            h = tanh_autograd(h);
        }
    }

    Tensor* y_pred = h;
    return y_pred;
}

// Tensor* forward_old(Linear* l1, Linear* l2, Linear* l3, Tensor* x) {
//     Tensor* h = linear_forward(l1, x);
//     h = tanh_autograd(h);
//     h = linear_forward(l2, h);
//     h = tanh_autograd(h);
//     h = linear_forward(l3, h);
//     Tensor* y_pred = tanh_autograd(h);

//     return y_pred;
// }


int main() {

Params p = {0};

load_params("params.txt", &p);

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
int num_layers = p.num_layers;// 3;
int hidden_size = p.hidden_layer;
Linear** linear_layer_list = malloc(num_layers*sizeof(Linear*));
int in_feat, out_feat;
for (int i =0; i<num_layers; i++) {
    in_feat = hidden_size; 
    out_feat= hidden_size;
    if(i==0){
        in_feat = 1;// hidden_size;
    }
    if (i==num_layers-1) {
        out_feat = 1;
    }
    Linear* t = linear_create(model, in_feat, out_feat, DEVICE_CUDA);
    linear_layer_list[i] = t;// linear_create(model, 1, 64, DEVICE_CUDA);
}

float lr = p.lr;// 0.03;
int epochs = p.epochs;// 2000;
FILE *fp = fopen("loss_data.csv", "w");
    if (!fp) {
        perror("Failed to open file");
        return 1;
    }

    fprintf(fp, "epoch,loss\n");  // CSV header


printf("entered for loop.\n");
int m = epochs - 1;
for (int epoch = 0; epoch < epochs; ++epoch) {

        printf("\n\nForward-------------------------------------------->\n\n");
        printf("\nforward: ");
        Tensor* y_pred = forward(linear_layer_list, num_layers, x);//(l1, l2, l3, x);      // on CUDA
        Tensor* loss   = MSE(y_pred, y_true);
        model_zero_grad(model);
        int shape[1] = {1};
        Tensor* root_grad = tensor_ones(shape, 1, 0, loss->device);
        
        tensor_backward(loss, root_grad->data);   // assume NULL means grad=1 for scalar
        printf("\n");
        model_sgd_step(model, lr);

        printf("\n\nBackward-------------------------------------------->\n\n");

        Tensor* t1 = tensor_to_cpu(loss);

        

        t1 = loss;
        /*
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
            printf("\ntanh2 (input): *************************************** \n");
            t2 = tensor_to_cpu(t1->parents[0]->parents[0]->parents[0]->parents[0]->parents[0]);
            print_tensor_info(t2);
            printf("\nlinear2 b: *************************************** \n");
            t2 = tensor_to_cpu(t1->parents[0]->parents[0]->parents[0]->parents[0]->parents[0]->parents[1]);
            print_tensor_info(t2);
            printf("\nlinear2 matmul (input): *************************************** \n");
            t2 = tensor_to_cpu(t1->parents[0]->parents[0]->parents[0]->parents[0]->parents[0]->parents[0]->parents[0]);
            print_tensor_info(t2);
            printf("\nlinear2 W: *************************************** \n");
            t2 = tensor_to_cpu(l2->W);
            print_tensor_info(t2);
            printf("\nrelu (input): *************************************** \n");
            t2 = tensor_to_cpu(t1->parents[0]->parents[0]->parents[0]->parents[0]->parents[0]->parents[0]->parents[0]->parents[0]);
            print_tensor_info(t2);

            printf("\nlinear1 b: *************************************** \n");
            t2 = tensor_to_cpu(t1->parents[0]->parents[0]->parents[0]->parents[0]->parents[0]->parents[0]->parents[0]->parents[0]->parents[1]);
            print_tensor_info(t2);
            printf("\nlinear1 matmul (input): *************************************** \n");
            t2 = tensor_to_cpu(t1->parents[0]->parents[0]->parents[0]->parents[0]->parents[0]->parents[0]->parents[0]->parents[0]->parents[0]->parents[0]);
            print_tensor_info(t2);
            printf("\nlinear1 W: *************************************** \n");
            t2 = tensor_to_cpu(l1->W);
            print_tensor_info(t2);

            

            // t1 = t1->parents[0];
            // }
            // break;
        }
        */
        
        printf("\nepoch: %d\n", epoch);
        print_tensor_info(loss);
        printf("\nss_cpu is: ");
        Tensor* loss_cpu = tensor_to_cpu(loss);
        printf("-->%f\n", loss_cpu->data[0]);
        fprintf(fp, "%d,%f\n", epoch, loss_cpu->data[0]);

        free_tensor(y_pred);
        free_tensor(loss);
    }

    Tensor* y_true_cpu = tensor_to_cpu(y_true);
    printf("\n Here is the y_true : \n");
    print_tensor_info(y_true_cpu);
    
    Tensor* y_pred = forward(linear_layer_list, num_layers, x);
    Tensor* y_cpu = tensor_to_cpu(y_pred);
    printf("\n Here is the y_cpu : \n");
    print_tensor_info(y_cpu);
    fclose(fp);






    // Call Python to plot
    int ret = system("python ../py/plot_loss.py");
    if (ret != 0) {
        fprintf(stderr, "Failed to run Python script\n");
    }

    // Write predictions and ground truth to file
    FILE *out = fopen("plot_data.txt", "w");
    if (!out) {
        perror("plot_data.txt");
        exit(1);
    }

    // Assuming both tensors are 1-D of same length
    int N = y_true_cpu->size;  // modify if your tensor stores size differently

    for (int i = 0; i < N; i++) {
        float yt = y_true_cpu->data[i];
        float yp = y_cpu->data[i];
        fprintf(out, "%f %f\n", yt, yp);
    }

    fclose(out);

    // Call Python to plot
    ret = system("python ../py/plot_gt_vs_pred.py");
    if (ret != 0) {
        fprintf(stderr, "Failed to run Python script\n");
    }







    // Cleanup
    free_tensor(x);
    free_tensor(y_true);
    // linear_free(l1);
    // linear_free(l2);
    model_free(model);





return 0;
}














