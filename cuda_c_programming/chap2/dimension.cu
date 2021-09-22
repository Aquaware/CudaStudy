#include "../common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


__global__
void check(void){
	printf("threadIdx: (%d, %d, %d) blockIdx: (%d, %d, %d) blockDim: (%d, %d, %d) gridDim: (%d, %d, %d)\n",
				threadIdx.x, threadIdx.y, threadIdx.y,
				blockIdx.x, blockIdx.y, blockIdx.z,
				blockDim.x, blockDim.y, blockDim.z,
				gridDim.x, gridDim.y, gridDim.z);
}

int main(int argc, char** argv) {
	int n = 16;
	dim3 block(16);
	dim3 grid((n + block.x - 1) / block.x);

	printf("grid : (%d, %d, %d) \n", grid.x, grid.y, grid.z);
	printf("block :  (%d, %d, %d) \n", block.x, block.y, block.z);
	check <<<grid, block >>> ();
	cudaDeviceReset();

}

