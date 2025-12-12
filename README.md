# CPytorch
My own implementation of Pytorch

![Training Progress](py/training_progress.gif)


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
