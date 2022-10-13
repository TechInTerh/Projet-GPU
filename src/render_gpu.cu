#include "render_gpu.cuh"
#include <iostream>
#include "cuda_runtime_api.h"
__global__ void kernel()
{
	printf("Hello from GPU!");
}

void use_gpu()
{
	kernel<<<1,1>>>();
	cudaDeviceSynchronize();
}
