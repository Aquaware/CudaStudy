#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

extern "C"
{
	void runAdd(dim3 grid, dim3 block, float* a, float* b, float* c, int size);
}

