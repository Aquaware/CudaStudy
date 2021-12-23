#include <iostream>
#include <stdlib.h>

#include <cuda_runtime.h>

#include "kernels.h"



using namespace std;

__global__
void addX(float* a, float* b, float* c)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	c[i] = a[i] + b[i];
}



void cu() {
	int size = 100;
	float* a = new float[size];
	float* b = new float[size];
	float* c = new float[size];

	for (int i = 0; i < size; i++) {
		a[i] = 1.0f;
		b[i] = 200.0f;
		c[i] = 0.0f;
	}

	int bytes = size * sizeof(float);

	float* dA;
	float* dB;
	float* dC;

	cudaError_t ret;
	ret = cudaMalloc(&dA, bytes);
	cout << "ret: " << ret << endl;
	ret = cudaMalloc(&dB, bytes);
	cout << "ret: " << ret << endl;
	ret = cudaMalloc(&dC, bytes);
	cout << "ret: " << ret << endl;

	ret = cudaMemcpy(dA, a, bytes, cudaMemcpyHostToDevice);
	cout << "ret: " << ret << endl;

	ret = cudaMemcpy(dB, b, bytes, cudaMemcpyHostToDevice);
	cout << "ret: " << ret << endl;

	ret = cudaMemcpy(dC, c, bytes, cudaMemcpyHostToDevice);
	cout << "ret: " << ret << endl;

	dim3 block = dim3(16);
	dim3 grid = dim3((size + block.x - 1) / block.x);
	addX<<<grid, block>>>(dA, dB, dC);

	ret = cudaDeviceSynchronize();
	cout << "ret: " << ret << endl;

	ret = cudaMemcpy(c, dC, bytes, cudaMemcpyDeviceToHost);
	cout << "ret: " << ret << endl;

	for (int i = 0; i < 5; i++) {
		cout << "a: " << a[i] << " b: " << b[i] << " c: " << c[i] << endl;
	}

	ret = cudaFree(dA);
	cout << "ret: " << ret << endl;

	ret = cudaFree(dB);
	cout << "ret: " << ret << endl;

	ret = cudaFree(dC);
	cout << "ret: " << ret << endl;

	delete[] a;
	delete[] b;
	delete[] c;
}
