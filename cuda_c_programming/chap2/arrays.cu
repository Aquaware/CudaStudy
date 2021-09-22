#if defined(_WIN32) || (_MSC_VER)
#define VC_MODE
#endif

#include "../common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void gpu(float*a, float*b, float*c, int n);

#ifdef VC_MODE
double cpuSecond(){
    _timeb tp;
    _ftime(&tp);
    return ((double)tp.time + (double)tp.millitm / 1000.0);
}
#else
double cpuSecond(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}
#endif



void sum(float *a, float* b, float* c, const int num) {
	for (int idx = 0; idx < num; idx++) {
		c[idx] = a[idx] + b[idx];
	}
}

__global__
void sumGpu(float *A, float *B, float *C, const int num)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    C[i] = A[i] + B[i];
}

void genData(float *ip, int size) {
	time_t t;
	srand((unsigned int) time(&t));
	for (int i = 0; i < size; i++) {
		ip[i] = (float)(rand() & 0xff) / 10.0f;
	}
	return;
}

void checkResult(float *hostRef, float *gpuRef, const int N){
    double epsilon = 1.0E-8;
    bool match = 1;
    for (int i = 0; i < N; i++){
        if (abs(hostRef[i] - gpuRef[i]) > epsilon){
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i],
                   gpuRef[i], i);
            break;
        }
    }
    if (match) printf("Arrays match.\n\n");
    return;
}

void setupGpu(int devNo) {
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, devNo));
	printf("*** Using Device %d : %s***\n", devNo, deviceProp.name);
	CHECK(cudaSetDevice(devNo))
}

int main(int argc, char** argv) {
	int n = 1024;
	size_t bytes = n * sizeof(float);
	float* a;
	float* b;
	float* ans1;
	float* ans2;

	a = (float*) malloc(bytes);
	b = (float*) malloc(bytes);
	ans1 = (float*) malloc(bytes);
	ans2 = (float*) malloc(bytes);
	memset(ans1, 0, bytes);
	memset(ans2, 0, bytes);

	genData(a, n);
	genData(b, n);
	sum(a, b, ans1, n);
	gpu(a, b, ans2, n);
	checkResult(ans1, ans2, n);

	free(a);
	free(b);
	free(ans1);
	free(ans2);
	return 0;
}

void gpu(float*a, float*b, float*c, int n) {
	setupGpu(0);
	size_t bytes = n * sizeof(float);

	float* da;
	float* db;
	float* dc;
	CHECK(cudaMalloc((float**)&da, bytes));
	CHECK(cudaMalloc((float**)&db, bytes));
	CHECK(cudaMalloc((float**)&dc, bytes));
	CHECK(cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(dc, c, bytes, cudaMemcpyHostToDevice));

	dim3 block(n);
	dim3 grid(1);

	double tBegin = cpuSecond();
	sumGpu <<<grid, block>>>(da, db, dc, n);
	CHECK(cudaDeviceSynchronize());
	printf("GPU process elapsed time : %f (sec)\n", cpuSecond() - tBegin);
	CHECK(cudaGetLastError());

	CHECK(cudaMemcpy(c, dc, bytes, cudaMemcpyDeviceToHost));

	CHECK(cudaFree(da));
	CHECK(cudaFree(db));
	CHECK(cudaFree(dc));
}
