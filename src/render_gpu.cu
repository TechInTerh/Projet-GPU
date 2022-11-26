#include "render_gpu.cuh"

#include <iostream>
#include "cuda_runtime_api.h"

__global__ void grayscale(uchar3 *matImg, size_t width, size_t height)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= width * height)
	{
		return;
	}
	uchar3 px_in = matImg[idx];
	char px_out = ceil(0.3 * px_in.x + 0.59 * px_in.y + 0.11 * px_in.z);
	uchar3 newVal = createUchar3(px_out, px_out, px_out);
	matImg[idx] = newVal;
}



void use_gpu(gil::rgb8_image_t &image)
{
	matrixImage<uchar3> *matImg = toMatrixImage(image);
	matImg->toGpu();


	size_t size_block= 1024;
	size_t size_grid = (matImg->width * matImg->height + size_block - 1) / size_block;
	//grayscale<<<1024,(matImg->width$la>>>(matImg->buffer, matImg->width, matImg->height);
	grayscale<<<size_grid,size_block>>>(matImg->buffer,matImg->width,matImg->height);
	cudaError_t err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
	{
		spdlog::error("Error in cudaDeviceSynchronize: {}", cudaGetErrorString(err));
		std::exit(1);
	}
	matImg->toCpu();
	write_image(matImg);
}
