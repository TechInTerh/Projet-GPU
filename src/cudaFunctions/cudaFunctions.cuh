#ifndef GPGPU_CUDAFUNCTIONS_CUH
#define GPGPU_CUDAFUNCTIONS_CUH

#include "env.cuh"

void *cudaMallocX(size_t size);

void *cudaMallocPitchX(size_t *pitch, size_t width, size_t height);

void cudaMemcpy2DX(void *dst, size_t dpitch, const void *src, size_t spitch,
				   size_t width, size_t height, cudaMemcpyKind kind);
void cudaFreeX(void *ptr);

void cudaDeviceSynchronizeX();

void cudaMemcpyX(void *dst, const void *src, size_t count, cudaMemcpyKind kind);

#endif //GPGPU_CUDAFUNCTIONS_CUH
