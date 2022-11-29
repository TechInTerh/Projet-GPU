#include "cudaFunctions.cuh"
#include "cuda.h"
#include "env.cuh"

// function  from cudaMemcpyKind to str
const char *cudaMemcpyKindToStr(cudaMemcpyKind kind)
{
	switch (kind)
	{
	case cudaMemcpyHostToHost:
		return "cudaMemcpyHostToHost";
	case cudaMemcpyHostToDevice:
		return "cudaMemcpyHostToDevice";
	case cudaMemcpyDeviceToHost:
		return "cudaMemcpyDeviceToHost";
	case cudaMemcpyDeviceToDevice:
		return "cudaMemcpyDeviceToDevice";
	default:
		return "cudaMemcpyDefault";
	}
}


void *cudaMallocX(size_t size)
{
	void *ret;
	cudaError_t err = cudaMalloc(&ret,size);
	if (err != cudaSuccess)
	{
		spdlog::error("Error allocating memory in GPU: {}", cudaGetErrorString(err));
		std::exit(1);
	}
	return ret;
}

void *cudaMallocPitchX(size_t *pitch, size_t width, size_t height)
{
	void *ret;
	cudaError_t err = cudaMallocPitch(&ret, pitch, width, height);
	if (err != cudaSuccess)
	{
		spdlog::error("Error allocating memory in GPU: {}", cudaGetErrorString(err));
		std::exit(1);
	}
	return ret;
}

void cudaMemcpy2DX(void *dst, size_t dpitch, const void *src, size_t spitch,
				   size_t width, size_t height, cudaMemcpyKind kind)
{
	cudaError_t err = cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind);
	if (err != cudaSuccess)
	{
		spdlog::error("Error copying memory in {}: {}", cudaMemcpyKindToStr(kind), cudaGetErrorString(err));
		std::exit(1);
	}
}

void cudaFreeX(void *ptr)
{
	cudaError_t err = cudaFree(ptr);
	if (err != cudaSuccess)
	{
		spdlog::error("Error freeing memory in GPU: {}", cudaGetErrorString(err));
		std::exit(1);
	}
}

void cudaDeviceSynchronizeX()
{
	cudaError_t err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
	{
		spdlog::error("Error synchronizing GPU: {}", cudaGetErrorString(err));
		std::exit(1);
	}
}

void cudaMemcpyX(void *dst, const void *src, size_t count, cudaMemcpyKind kind)
{
	cudaError_t err = cudaMemcpy(dst, src, count, kind);
	if (err != cudaSuccess)
	{
		spdlog::error("Error copying memory in {}: {}", cudaMemcpyKindToStr(kind), cudaGetErrorString(err));
		std::exit(1);
	}
}
