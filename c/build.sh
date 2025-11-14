#!/usr/bin/env bash
gcc src/tensor.c src/ops_add_sub.c src/ops_mul_div.c src/main.c -o main.exe
echo "Build finished."
