#if defined(_WIN32) || (_MSC_VER)
#define VC_MODE
#endif

#include "../common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void test();
void testGpu(int* matrix, int nx, int ny);
void add();
void addGpu(float* a, float* b, float* ans, int nx, int ny);

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

void setupGpu(int devNo) {
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, devNo));
	printf("*** Using Device %d : %s***\n", devNo, deviceProp.name);
	CHECK(cudaSetDevice(devNo))
}

int setHighestPerformanceGpu() {
	int num = 0;
	int device = 0;
	cudaGetDeviceCount(&num);
	if (num == 0) {
		return -1;
	}
	if(num > 1) {
		int max = 0;
		for (int i = 0; i < num; i++) {
			cudaDeviceProp props;
			cudaGetDeviceProperties(&props, i);
			if (max < props.multiProcessorCount) {
				max = props.multiProcessorCount;
				device = i;
			}
		}
	}
	cudaSetDevice(device);
	return device;
}

void gpuInfo(int deviceNo) {
	cudaDeviceProp prop;
	int driverVersion;
	int runtimeVersion;
	cudaGetDeviceProperties(&prop, deviceNo);
	cudaDriverGetVersion(&driverVersion);
	cudaRuntimeGetVersion(&runtimeVersion);
	printf("=======\n");
	printf("CUDA Device No: %d  Name: %s\n", deviceNo, prop.name);
	printf("  driver ver.%d, runtime ver.%d\n", driverVersion, runtimeVersion);
	printf("  Capability: %d.%d\n", prop.major, prop.minor);
	printf("  Global memory: %f Gbytes\n", (float) prop.totalGlobalMem /  pow(1024.0, 3));
	printf("  Clock rate: %0.2f GHz \n", prop.clockRate * 1e-6f);
	printf("  Bus width: %d bit\n", prop.memoryBusWidth);
	if (prop.l2CacheSize) {
		printf("  L2 Cache size: %d bytes \n", prop.l2CacheSize);
	}
	printf("  Max Texture Dimension Size (x, y, z) 1D: (%d)\n"
			"                                       2D: (%d, %d)\n"
			"                                       3D: (%d, %d, %d)\n",
			prop.maxTexture1D,prop.maxTexture2D[0], prop.maxTexture2D[1], prop.maxTexture3D[0], prop.maxTexture3D[1], prop.maxTexture3D[2]);
	printf("  Max Layered Texture   Size x Layer   1D: (%d) x %d\n"
			"                                       2D: (%d, %d) x %d\n",
			prop.maxTexture1DLayered[0], prop.maxTexture1DLayered[1],prop.maxTexture2DLayered[0], prop.maxTexture2DLayered[1], prop.maxTexture2DLayered[2]);
	printf("  Constant memory: %lu bytes\n", prop.totalConstMem);
	printf("  Shared memory per block: %lu bytes\n", prop.sharedMemPerBlock);
	printf("  Registers per block: %d bytes\n", prop.regsPerBlock);
	printf("  Warp size: %d \n", prop.warpSize);
	printf("  Max threads per multi processor: %d\n", prop.maxThreadsPerMultiProcessor);
	printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
	printf("  Max threads of each dimension of a block: %d x %d x%d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
	printf("  Max sizes of each dimension of grid: %d x %d x %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
	printf("  Max memory pitch: %lu bytes\n", prop.memPitch);
	printf("=======\n");
}

// --------------------------------------

void printMatrix(int *matrix, const int nx, const int ny) {
	int *p = matrix;
	printf("*\nMatrix: (%d, %d)\n", nx, ny);
	for (int y = 0; y < ny; y++) {
		for(int x = 0; x < nx; x++) {
			printf("%3d", p[x]);
		}
		p += nx;
		printf("*\n");
	}
	printf("*\n");
	return;
}

void addMatrix(float* a, float* b, float* c, const int nx, const int ny) {
    float *ia = a;
    float *ib = b;
    float *ic = c;

    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix = 0; ix < nx; ix++)
        {
            ic[ix] = ia[ix] + ib[ix];

        }

        ia += nx;
        ib += nx;
        ic += nx;
    }

    return;
}

__global__
void printMatrixGpu(int *matrix, const int nx, const int ny) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int index = y * nx + x;
	printf("threadIdx: (%d, %d) blockIdx: (%d, %d) corrdinate (%d, %d) global index: %2d ival: %2d\n", 	threadIdx.x, threadIdx.y,
																														blockIdx.x, blockIdx.y,
																														x, y,
																														index, matrix[index] );
	return;
}


__global__
void addMatrixGpu(float* a, float* b, float* c, const int nx, const int ny) {
	unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int index = x + y * nx;
	if (x < nx  && y < ny) {
		c[index] = a[index] + b[index];
	}
}


void randomData(float *ip, int size) {
	time_t t;
	srand((unsigned int) time(&t));
	for (int i = 0; i < size; i++) {
		ip[i] = (float)(rand() & 0xff) / 10.0f;
	}
	return;
}

void seriesData(float *p, int size) {
	for (int i = 0; i < size; i++) {
		p[i] = i;
	}
}

void compare(float *hostRef, float *gpuRef, const int n, bool should_all){
    double epsilon = 1.0E-8;
    int error = 0;
    int error_max = 5;
    for (int i = 0; i < n; i++){
        if (abs(hostRef[i] - gpuRef[i]) > epsilon){
        	error++;
        	if (!should_all) {
        		printf("Compare error! CPU: %5.2f GPU: %5.2f  at %d\n", hostRef[i], gpuRef[i], i);
        		if (error > error_max) {
        			break;
        		}
        	}
        }
    }

    printf("*** Compare Result ***  data size: %d, error count: %d\n", n, error);
    return;
}

int main(int argc, char** argv) {
	test();
	add();
	return 0;
}

void test() {
	int nx = 8;
	int ny = 6;
	int n = nx * ny;
	size_t bytes = n * sizeof(int);
	int* a;
	a = (int*) malloc(bytes);
	for (int i = 0; i < n; i++) {
		a[i] = i;
	}
	printMatrix(a, nx, ny);
	testGpu(a, nx, ny);
	free(a);
	return;
}

void testGpu(int* matrix, int nx, int ny) {
	setupGpu(0);

	size_t bytes = nx * ny * sizeof(int);

	int* da;
	CHECK(cudaMalloc((int**)&da, bytes));
	CHECK(cudaMemcpy(da, matrix, bytes, cudaMemcpyHostToDevice));

	dim3 block(4, 2);
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

	double tBegin = cpuSecond();
	printMatrixGpu <<<grid, block>>>(da, nx, ny);
	CHECK(cudaDeviceSynchronize());
	printf("GPU process elapsed time : %f (sec)\n", cpuSecond() - tBegin);
	CHECK(cudaGetLastError());
	CHECK(cudaFree(da));
}

void add() {
	int nx = 1 << 8;
	int ny = nx;

	printf("\n\nMatrix size: (%d, %d) \n", nx, ny);

	int n = nx * ny;
	size_t bytes = n * sizeof(float);
	float* a;
	float* b;
	float* ans1;
	float* ans2;

	a = (float*) malloc(bytes);
	b = (float*) malloc(bytes);
	ans1 = (float*) malloc(bytes);
	ans2 = (float*) malloc(bytes);

	randomData(a, n);
	randomData(b, n);

	memset(ans1, 0, bytes);
	memset(ans2, 0, bytes);

	addMatrix(a, b, ans1, nx, ny);
	addGpu(a, b, ans2, nx, ny);
	compare(ans1, ans2, n, true);

	free(a);
	free(b);
	free(ans1);
	free(ans2);

}

void addGpu(float* a, float* b, float* ans, int nx, int ny) {
	//setupGpu(0);
	auto device = setHighestPerformanceGpu();
	//printf("Device No: %d\n", device);
	gpuInfo(device);

	size_t bytes = nx * ny * sizeof(float);

	float* da;
	float* db;
	float* dc;
	CHECK(cudaMalloc((void**)&da, bytes));
	CHECK(cudaMalloc((void**)&db, bytes));
	CHECK(cudaMalloc((void**)&dc, bytes));
	CHECK(cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice));

	dim3 block(32, 32);
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

	double tBegin = cpuSecond();
	addMatrixGpu <<<grid, block>>>(da, db, dc, nx, ny);
	//sumMatrixOnGPU2D <<<grid, block>>>(da, db, dc, nx, ny);
	CHECK(cudaDeviceSynchronize());
	printf("GPU process elapsed time : %f (sec)\n", cpuSecond() - tBegin);
	CHECK(cudaGetLastError());
	CHECK(cudaMemcpy(ans, dc, bytes, cudaMemcpyDeviceToHost));

	CHECK(cudaFree(da));
	CHECK(cudaFree(db));
	CHECK(cudaFree(dc));
}
