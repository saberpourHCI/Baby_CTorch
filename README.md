# CPytorch
My own implementation of Pytorch




open "x64 Native Tools Command Prompt for VS22", run the following:

'''
cd D:\projects\ctorch\CPytorch\c
code .
'''

Then in the VS code that opens, open a "command prompt" terminal and run the following:

'''
nvcc src/tensor.c src/cuda_utils.cu src/ops_add_sub_cpu.c src/ops_add_sub_cuda.cu src/ops_add_sub.c src/ops_mul_div_cpu.c src/ops_mul_div_cuda.cu src/ops_mul_div.c src/ops_matmul.c src/ops_matmul_cpu.c src/ops_matmul_cuda.cu src/linear.c src/activation_cpu.c src/activation_cuda.cu src/activation.c src/params.cu src/model.c src/loss.c src/main_training_1.c -o main1.exe
'''

This would generate the "main.exe" file, that you can execute later.

















run the following in Git bash:

'''
chmod +x build.sh     # only once
./build.sh
./main.exe

'''