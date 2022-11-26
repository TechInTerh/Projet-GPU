#include "render_gpu.cuh"

#include <iostream>
#include "cuda_runtime_api.h"
__global__ void kernel(uchar3 *matImg)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	//create a zero uchar3
	uchar3 val = createUchar3(255, 0,0);

	matImg[idx] = val;
}

void use_gpu(gil::rgb8_image_t &image)
{
	matrixImage<uchar3> *matImg = toMatrixImage(image);
	matImg->toGpu();

	kernel<<<255, 1>>>(matImg->buffer);
	cudaError_t err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
	{
		std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
	}
	matImg->toCpu();
	write_image(matImg);
}
