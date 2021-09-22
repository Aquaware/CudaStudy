#include "gpu_common.h"
#include <sys/time.h>

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
	CHECK(cudaSetDevice(devNo));
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
	setupGpu(device);
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
