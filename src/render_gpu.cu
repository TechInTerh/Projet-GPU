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

/*
__global__ void
pxGaussianBlur(uchar3 *matImg, uchar3 *matOut, size_t width, size_t height,
			   size_t kernel_size)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= width * height)
	{
		return;
	}
	uchar3 px_in = matImg[idx];
	size_t new_x = 0;
	size_t new_y = 0;
	size_t new_z = 0;
	float kernel_value = 1;
	for (int k_w = 0; k_w < kernel_size; k_w++)
	{
		for (int k_h = 0; k_h < kernel_size; k_h++)
		{
			size_t cur_loc =
					idx + (k_h - (size_t) ceil(kernel_size / 2)) * width +
					(k_w - (size_t) ceil(kernel_size / 2));
			if (cur_loc >= width * height)
			{
				continue;
			}
			uchar3 px_tmp = matImg[cur_loc];
			for (int i = 0; i < 3; i++)
			{
				new_x += px_tmp.x * kernel_value;
				new_y += px_tmp.y * kernel_value;
				new_z += px_tmp.z * kernel_value;
			}
		}
	}
	for (int i = 0; i < 3; i++)
	{
		new_x /= kernel_size * kernel_size;
		new_y /= kernel_size * kernel_size;
		new_z /= kernel_size * kernel_size;
		px_in.x = (char) new_x;
		px_in.y = (char) new_y;
		px_in.z = (char) new_z;
	}

	matOut[idx] = px_in;
}
*/

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
	/*
	matrixImage<uchar3> *matOut = matImg->deepCopy();
	pxGaussianBlur<<<size_grid, size_block>>>(matImg->buffer, matOut->buffer,
											  matImg->width, matImg->height,1);

	err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
	{
		spdlog::error("Error in cudaDeviceSynchronize: {}",
					  cudaGetErrorString(err));
		std::exit(1);
	}
	 */
	matImg->toCpu();
	spdlog::info("Copying on CPU");
	//matOut->toCpu();

	write_image(matImg);
	delete matImg;
	//delete matOut;
}
