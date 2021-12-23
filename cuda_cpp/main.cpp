/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* Example of integrating CUDA functions into an existing
 * application / framework.
 * CPP code representing the existing application / framework.
 * Compiled with default CPP compiler.
 */

// includes, system
#include <iostream>
#include <stdlib.h>

#include <cuda_runtime.h>

#include "kernels.h"

#include "process.cuh"

using namespace std;


void cpp() {
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
	runAdd(grid, block, dA, dB, dC, size);

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


int main(int argc, char **argv)
{
	cpp();
	return 0;
}
