#include <stdio.h>
#include <math.h>

#include "kernels.h"

__global__
void add(float* a, float* b, float* c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];
}

void runAdd(dim3 grid, dim3 block, float* a, float* b, float* c, int size) {
    add << <grid, block >> > (a, b, c);
}


