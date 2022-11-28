#include "render_gpu.cuh"

#include <iostream>
#include "cuda_runtime_api.h"

// Computes the pointer address of a given value in a 2D array given:
// baseAddress: the base address of the buffer
// col: the col coordinate of the value
// row: the row coordinate of the value
// pitch: the actual allocation size **in bytes** of a row plus its padding
template<typename T>
__device__ T *eltPtr(T *baseAddress, size_t col, size_t row, size_t pitch)
{
	return (T *) ((char *) baseAddress + row * pitch +
				  col * sizeof(T));  // FIXME
}

__global__ void
grayscale(uchar3 *matImg, size_t width, size_t height, size_t pitch)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t idy = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx > width || idy > height)
	{
		return;
	}
	uchar3 *px_in = eltPtr<uchar3>(matImg, idx, idy, pitch);
	char px_out = ceil(0.3 * px_in->x + 0.59 * px_in->y + 0.11 * px_in->z);
	uchar3 newVal = createUchar3(px_out, px_out, px_out);
	*px_in = newVal;
}

__global__ void
gaussianBlur(uchar3 *matIn, uchar3 *matOut, size_t width, size_t height,
			 size_t pitch)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx > width || idy > height)
	{
		return;
	}
	const size_t kernel_size = 3;
	const float ker[kernel_size][kernel_size] = {{0.0625, 0.125, 0.0625},
												 {0.125,  0.25,  0.125},
												 {0.0625, 0.125, 0.0625}};
	uchar3 *px_out = eltPtr<uchar3>(matOut, idx, idy, pitch);
	float px_x = 0;
	float px_y = 0;
	float px_z = 0;
	for (size_t k_w = 0; k_w < kernel_size; k_w++)
	{
		for (size_t k_h = 0; k_h < kernel_size; k_h++)
		{
			if (idx + k_w - 1 < width && idy + k_h - 1 < height)
			{
				uchar3 *px_in = eltPtr<uchar3>(matIn, idx + k_w - 1,
											   idy + k_h - 1, pitch);
				px_x += px_in->x * ker[k_w][k_h];
				px_y += px_in->y * ker[k_w][k_h];
				px_z += px_in->z * ker[k_w][k_h];
			}
		}
	}
	uchar3 newVal = createUchar3((char) px_x, (char) px_y, (char) px_z);
	*px_out = newVal;

}

void use_gpu(gil::rgb8_image_t &image)
{
	matrixImage<uchar3> *matImg = toMatrixImage(image);
	matImg->toGpu();


	dim3 threads(32, 32);
	dim3 blocks((matImg->width + threads.x - 1) / threads.x,
				(matImg->height + threads.y - 1) / threads.y);
	spdlog::info("Lunching Grayscale");
	grayscale<<<blocks, threads>>>(matImg->buffer, matImg->width,
								   matImg->height, matImg->pitch);
	cudaError_t err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
	{
		spdlog::error("Error in cudaDeviceSynchronize: {}",
					  cudaGetErrorString(err));
		std::exit(1);
	}
	spdlog::info("Lunching Gaussian Blur");
	matrixImage<uchar3> *matOut = matImg->deepCopy();
	for (int i = 0; i < 20; i++)
	{
		gaussianBlur<<<blocks, threads>>>(matImg->buffer, matOut->buffer,
										  matImg->width, matImg->height,
										  matImg->pitch);
		cudaError_t err = cudaDeviceSynchronize();
		if (err != cudaSuccess)
		{
			spdlog::error("Error in cudaDeviceSynchronize: {}",
						  cudaGetErrorString(err));
			std::exit(1);
		}
		matrixImage<uchar3> *tmp = matImg;
		matImg = matOut;
		matOut = tmp;
	}

	matOut->toCpu();
	spdlog::info("Copying on CPU");
	write_image(matOut);
	delete matImg;
	delete matOut;
}
