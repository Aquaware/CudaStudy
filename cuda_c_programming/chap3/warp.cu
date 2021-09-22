#include "gpu_common.h"
#include <cuda_runtime.h>

void test(int dataSize, int blockSize);

__global__
void kernel1(float* c) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	float a = 0.0f;
	float b = 0.0f;
	if (index %2 == 0) {
		a = 100.0f;
	} else {
		b = 200.0f;
	}
	c[index] = a + b;
}

__global__
void kernel2(float* c) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	float a = 0.0f;
	float b = 0.0f;
	if ((index / warpSize) %2 == 0) {
		a = 100.0f;
	} else {
		b = 200.0f;
	}
	c[index] = a + b;
}

__global__
void warmingup(float *c){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;
    if ((tid / warpSize) % 2 == 0){
        ia = 100.0f;
    } else {
        ib = 200.0f;
    }
    c[tid] = ia + ib;
}


int main(int argc, char** argv) {
	int dataSize = 64;
	int blockSize = 64;
	if (argc > 1) {
		blockSize = atoi(argv [1]);
	}
	if (argc > 2) {
		dataSize = atoi(argv[2]);
	}
	test(dataSize, blockSize);
	return 0;
}

void test(int dataSize, int blockSize) {
	auto device = setHighestPerformanceGpu();

	dim3 block(blockSize, 1);
	dim3 grid((dataSize + block.x - 1) / block.x, 1);
	printf("Block: %d, Grid: %d\n", block.x, grid.x);

	size_t bytes = dataSize * sizeof(float);
	float* d;
	CHECK(cudaMalloc((float**)&d, bytes));

	// Warming up
	CHECK(cudaDeviceSynchronize());
	warmingup<<<grid,block>>>(d);
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaGetLastError());

	size_t tbegin = cpuSecond();
	kernel1 <<<grid, block>>> (d);
	CHECK(cudaDeviceSynchronize());
	printf("kernel1 <<< %d, %d >>> elapsed %f sec \n", grid.x, block.x, cpuSecond() - tbegin);
	CHECK(cudaGetLastError());

	CHECK(cudaFree(d));
	CHECK(cudaDeviceReset());
}

