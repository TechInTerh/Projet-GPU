#include "cudaFunctions.cuh"
#include "cuda.h"
#include "env.cuh"
void *cudaMallocX(size_t size)
{
	void *ret;
	cudaMalloc(&ret,size);
	return ret;
}
