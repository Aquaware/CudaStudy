#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

double cpuSecond();
void setupGpu(int devNo);
int setHighestPerformanceGpu();
void gpuInfo(int deviceNo);

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
    }                                                                          \
}

