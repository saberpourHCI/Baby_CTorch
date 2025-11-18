# CPytorch
My own implementation of Pytorch




open "x64 Native Tools Command Prompt for VS22", run the following:

'''
cd D:\projects\ctorch\CPytorch\c
code .
'''

Then in the VS code that opens, open a "command prompt" terminal and run the following:

'''
nvcc src/tensor.c src/ops_add_sub_cpu.c src/ops_add_sub_cuda.cu src/ops_add_sub.c src/ops_mul_div.c src/main.c -o main.exe
'''

This would generate the "main.exe" file, that you can execute later.

















run the following in Git bash:

'''
chmod +x build.sh     # only once
./build.sh
./main.exe

'''