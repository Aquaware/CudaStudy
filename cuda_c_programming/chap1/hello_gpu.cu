#include <stdio.h>

__global__ void hello_gpu() {
	printf("Hell GPU thread ID: %d\n", threadIdx.x);
}

int main(int argc, char** argv) {
	hello_gpu<<<1, 10>>>();
	//cudaDeviceReset();
	cudaDeviceSynchronize();
	return 0;
}
