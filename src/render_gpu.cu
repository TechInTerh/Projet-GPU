#include "render_gpu.cuh"

#include <iostream>
#include "cuda_runtime_api.h"

// Computes the pointer address of a given value in a 2D array given:
// baseAddress: the base address of the buffer
// col: the col coordinate of the value
// row: the row coordinate of the value
// pitch: the actual allocation size **in bytes** of a row plus its padding
template<typename T>
__device__ __host__ T *
eltPtr(T *baseAddress, size_t col, size_t row, size_t pitch)
{
	return (T *) ((char *) baseAddress + row * pitch +
				  col * sizeof(T));  // FIXME
}

__global__ void
grayscale(uchar3 *matImg, float *matOut, size_t width, size_t height,
		  size_t pitch)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t idy = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx > width || idy > height)
	{
		return;
	}
	uchar3 *px_in = eltPtr<uchar3>(matImg, idx, idy, pitch);
	float val_out = 0.3 * px_in->x + 0.59 * px_in->y + 0.11 * px_in->z;
	float *px_out = eltPtr<float>(matOut, idx, idy, width * sizeof(float));
	*px_out = val_out;
}

__global__ void
gaussianBlur(float *matIn, float *matOut, size_t width, size_t height,
			 size_t pitch, size_t kernel_size, float *kernel,
			 size_t kernel_pitch)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
	size_t offset = kernel_size / 2;
	if (idx > width || idy > height)
	{
		return;
	}
	float val_out = 0;
	for (size_t k_w = 0; k_w < kernel_size; k_w++)
	{
		for (size_t k_h = 0; k_h < kernel_size; k_h++)
		{
			if (idx + k_w - offset < width && idy + k_h - offset < height)
			{
				float *px_in = eltPtr<float>(matIn, idx + k_w - offset,
											 idy + k_h - offset, pitch);
				val_out += *px_in * (*eltPtr(kernel, k_w, k_h, kernel_pitch));
			}
		}
	}
	float *px_out = eltPtr<float>(matOut, idx, idy, pitch);
	*px_out = val_out;

}

matrixImage<float> *generateKernelGPU(size_t kernel_size)
{
	matrixImage<float> *kernel = new matrixImage<float>(kernel_size,
														kernel_size);
	float mean = kernel_size / 2;
	float sigma = 1.0;
	float sum = 0.0;
	for (size_t y = 0; y < kernel_size; y++)
	{
		for (size_t x = 0; x < kernel_size; x++)
		{
			float val = std::exp(
					-0.5 * (std::pow((x - mean) / sigma, 2.0) +
							std::pow((y - mean) / sigma, 2.0))) /
						(2 * M_PI * sigma * sigma);
			kernel->buffer[y * kernel_size + x] = val;
			sum += val;
		}
	}
	for (size_t i = 0; i < kernel_size; i++)
	{
		for (size_t j = 0; j < kernel_size; j++)
		{
			kernel->buffer[i * kernel_size + j] /= sum;

		}
	}
	kernel->toGpu();
	return kernel;
}

void use_gpu(gil::rgb8_image_t &image)
{
	matrixImage<uchar3> *matImg = toMatrixImage(image);
	matImg->toGpu();


	dim3 threads(32, 32);
	dim3 blocks((matImg->width + threads.x - 1) / threads.x,
				(matImg->height + threads.y - 1) / threads.y);
	spdlog::info("Lunching Grayscale");
	matrixImage<float> *matGray = new matrixImage<float>(matImg->width,
														 matImg->height);
	matGray->toGpu();
	grayscale<<<blocks, threads>>>(
			matImg->buffer, matGray->buffer,
			matImg->width, matImg->height, matImg->pitch);
	cudaDeviceSynchronizeX();
	delete matImg;


	spdlog::info("Lunching Gaussian Blur");
	matrixImage<float> *matBlur = matGray->deepCopy();
	matrixImage<float> *kernel = generateKernelGPU(7);
	for (int i = 0; i < 60; i++)
	{
		gaussianBlur<<<blocks, threads>>>(matGray->buffer, matBlur->buffer,
										  matGray->width, matGray->height,
										  matGray->pitch, kernel->width,
										  kernel->buffer, kernel->pitch);
		cudaDeviceSynchronizeX();
		matrixImage<float> *tmp = matBlur;
		matBlur = matGray;
		matGray = tmp;
	}

	spdlog::info("Copying on CPU");
	matBlur->toCpu();
	matImg = matFloatToMatUchar3(matBlur);
	spdlog::info("Writing image");
	write_image(matImg, "img.png");
	delete matBlur;
	delete matImg;
	delete matGray;
	delete kernel;
}
