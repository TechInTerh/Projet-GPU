#include "render_gpu.cuh"

#include <iostream>
#include "cuda_runtime_api.h"

void write_image_float(matrixImage<float> *mat, std::string path)
{
	mat->toCpu();
	matrixImage<uchar3> *tmp = matFloatToMatUchar3(mat);
	write_image(tmp, path.c_str());
	delete tmp;
	mat->toGpu();

}

__device__ float my_max(float a, float b)
{
	return a > b ? a : b;
}

__device__ float my_min(float a, float b)
{
	return a < b ? a : b;
}

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
		  size_t pitch_in, size_t pitch_out)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t idy = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx > width || idy > height)
	{
		return;
	}
	uchar3 *px_in = eltPtr<uchar3>(matImg, idx, idy, pitch_in);
	float val_out = 0.3 * px_in->x + 0.59 * px_in->y + 0.11 * px_in->z;
	float *px_out = eltPtr<float>(matOut, idx, idy, pitch_out);
	*px_out = val_out;
}

__global__ void
gaussianBlur(float *matIn, float *matOut, size_t width, size_t height,
			 size_t pitch_in, size_t pitch_out, size_t kernel_size,
			 float *kernel,
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
											 idy + k_h - offset, pitch_in);
				val_out += *px_in * (*eltPtr(kernel, k_w, k_h, kernel_pitch));
			}
		}
	}
	float *px_out = eltPtr<float>(matOut, idx, idy, pitch_out);
	*px_out = my_min(val_out,255.f);

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

__global__ void
abs_diff(float *matIn, float *matOut, size_t width, size_t height,
		 size_t pitch)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx > width || idy > height)
	{
		return;
	}
	float *px_in = eltPtr<float>(matIn, idx, idy, pitch);
	float *px_out = eltPtr<float>(matOut, idx, idy, pitch);
	*px_out = std::abs(*px_in - *px_out);
}

matrixImage<float> *
grayBlur(gil::rgb8_image_t &image, size_t numberBlur, dim3 threads, dim3 blocks,
		 const char *name)
{
	matrixImage<uchar3> *matImg = toMatrixImage(image);
	matImg->toGpu();

	spdlog::info("Lunching Grayscale");
	matrixImage<float> *matGray = new matrixImage<float>(matImg->width,
														 matImg->height);
	matGray->toGpu();
	grayscale<<<blocks, threads>>>(
			matImg->buffer, matGray->buffer,
			matImg->width, matImg->height, matImg->pitch, matGray->pitch);
	cudaDeviceSynchronizeX();
	write_image_float(matGray, name);

	spdlog::info("Lunching Gaussian Blur");
	matrixImage<float> *matBlur = matGray->deepCopy();
	matrixImage<float> *kernel = generateKernelGPU(7);
	for (int i = 0; i < numberBlur; i++)
	{
		gaussianBlur<<<blocks, threads>>>(matGray->buffer, matBlur->buffer,
										  matGray->width, matGray->height,
										  matGray->pitch, matBlur->pitch,
										  kernel->width,
										  kernel->buffer, kernel->pitch);
		cudaDeviceSynchronizeX();
		if (i != numberBlur - 1)
		{
			matrixImage<float> *tmp = matGray;
			matGray = matBlur;
			matBlur = tmp;
		}
	}

	delete matImg;
	delete matGray;
	delete kernel;
	return matBlur;
}

/**
 * @brief Compute the absolute difference between two matrix, stocking the result in "@param matBlur2"
 * @param matBlur1
 * @param matBlur2
 * @param threads
 * @param blocks
 */
void lunch_abs_diff(matrixImage<float> *matBlur1, matrixImage<float> *matBlur2,
					dim3 threads, dim3 blocks)
{
	spdlog::info("Lunching abs diff");
	abs_diff<<<blocks, threads>>>(matBlur1->buffer, matBlur2->buffer,
								  matBlur2->width, matBlur2->height,
								  matBlur2->pitch);
	cudaDeviceSynchronizeX();
}


__global__ void dilatationErosion(float *matIn, float *matOut, size_t width,
								  size_t height, size_t pitch_in,size_t pitch_out, size_t se_w,
								  size_t se_h, bool d_or_e)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx > width || idy > height)
	{
		return;
	}
	float max_value = 0.f;
	float min_value = 255.f;
	size_t off_w = se_w / 2;
	size_t off_h = se_h / 2;
	for (size_t sw = 0; sw < se_w; sw++)
	{
		for (size_t sh = 0; sh < se_h; sh++)
		{
			if (idx + sw - off_w < width && idy + sh - off_h < height)
			{
				float *px_in = eltPtr<float>(matIn, idx + sw - off_w,
											 idy + sh - off_h, pitch_in);
				if (d_or_e)
					max_value = my_max(max_value, *px_in);

				else
					min_value = my_min(min_value, *px_in);

			}
		}
	}
	float *px_out = eltPtr<float>(matOut, idx, idy, pitch_out);
	if (d_or_e)
		*px_out = max_value;
	else
		*px_out = min_value;
}


void generate_histo(matrixImage<float> *mat_in, int *histo)
{
	for (size_t w = 0; w < mat_in->width; w++)
	{
		for (size_t h = 0; h < mat_in->height; h++)
		{
			float *tmp_px = mat_in->at(w, h);
			int value = (int) (*tmp_px + 0.5);
			histo[value] += 1;
		}
	}
}
__global__ void generate_histogram(float *matIn, size_t width, size_t height,
								   size_t pitch, int *histogram,
								   size_t hist_size)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx > width || idy > height)
	{
		return;
	}
	float *px_in = eltPtr<float>(matIn, idx, idy, pitch);
	int value = (int)floor(*px_in);
	atomicAdd(&histogram[value], 1);
}

int find_mean_intensity(int *histo, size_t nb_px)
{
	float sigmas[256] = {0.f};
	for (int i = 1; i < 256; i++)
	{
		float wb, wf, mu_b, mu_f, count_b;
		wb = wf = mu_b = mu_f = count_b = 0.f;
		for (int  j = 0; j < i; j++)
		{
			count_b += histo[j];
			mu_b += histo[j] * j;
		}
		wb = count_b / (float)nb_px;
		wf = 1.f - wb;
		mu_b /= count_b;
		for (int j = i; j < 256; j++)
		{
			mu_f += histo[j] * j;
		}
		mu_f /= (nb_px - count_b);
		sigmas[i] = wb * wf * std::pow(mu_b - mu_f, 2);
	}
	int max = 0;
	for (int i = 0; i < 256; i++)
	{
		if (sigmas[i] > sigmas[max])
			max = i;
	}
	return max;
}

__global__ void thresholding(float *matIn, size_t width, size_t height,
							 size_t pitch, int threshold)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx > width || idy > height)
	{
		return;
	}
	float *px_in = eltPtr<float>(matIn, idx, idy, pitch);
	if (*px_in >= threshold)
		*px_in = 255.f;
	else
		*px_in = 0.f;
}

void launchThreshold(matrixImage<float> *matIn, dim3 threads, dim3 blocks)
{
	spdlog::info("Lunching threshold");
	size_t size_histo = 256;
	int histo[256] = {0};
	int *gpu_histo = (int *) cudaMallocX(size_histo * sizeof(int));
	cudaMemcpyX(gpu_histo, histo, size_histo * sizeof(int), cudaMemcpyHostToDevice);

	generate_histogram<<<blocks, threads>>>(matIn->buffer, matIn->width,
											matIn->height, matIn->pitch, gpu_histo,
											size_histo);
	cudaDeviceSynchronizeX();
	cudaMemcpyX(histo, gpu_histo, size_histo * sizeof(int),
			   cudaMemcpyDeviceToHost);

	int sum = 0;
	for (int i = 0; i < 256; i++)
	{
		sum += histo[i];
	}
	spdlog::info("Sum : {}", sum);
	int mean = find_mean_intensity(histo, matIn->width * matIn->height);

	spdlog::info("Mean intensity is {}", mean);
	thresholding<<<blocks,threads>>>(matIn->buffer, matIn->width, matIn->height,
									 matIn->pitch, mean);
	cudaDeviceSynchronizeX();
	cudaFreeX(gpu_histo);
}



void launchMorphOpeningClosing(matrixImage<float> *matIn, dim3 threads,
							   dim3 blocks)
{
	/*
	size_t size1_w = 0.02 * matIn->width;
	size_t size1_h = 0.02 * matIn->height;
	size_t size2_w = 0.05 * matIn->width;
	size_t size2_h = 0.05 * matIn->height;
	*/
	size_t size1_w = 20;
	size_t size1_h = 20;
	size_t size2_w = 20;
	size_t size2_h = 20;

	spdlog::info("Lunching morph closing");
	matrixImage<float> *matOut = matIn->deepCopy();
	dilatationErosion<<<blocks, threads>>>(matIn->buffer, matOut->buffer,
										   matIn->width, matIn->height,
										   matIn->pitch,matOut->pitch, size1_w, size1_h,
										   true);
	cudaDeviceSynchronizeX();
	dilatationErosion<<<blocks, threads>>>(matOut->buffer, matIn->buffer,
										   matIn->width, matIn->height,
										   matIn->pitch,matOut->pitch, size1_w, size1_h,
										   false);
	cudaDeviceSynchronizeX();
	write_image_float(matIn, "gpu_morph_closing.png");
	spdlog::info("Lunching morph opening");
	dilatationErosion<<<blocks, threads>>>(matIn->buffer, matOut->buffer,
										   matIn->width, matIn->height,
										   matIn->pitch,matOut->pitch, size2_w, size2_h,
										   false);

	cudaDeviceSynchronizeX();
	dilatationErosion<<<blocks, threads>>>(matOut->buffer, matIn->buffer,
										   matIn->width, matIn->height,
										   matIn->pitch,matOut->pitch, size2_w, size2_h,
										   true);
	cudaDeviceSynchronizeX();
	delete matOut;
}
void use_gpu(gil::rgb8_image_t &image, gil::rgb8_image_t &image2)
{
	dim3 threads(32, 32);
	dim3 blocks((image.width() + threads.x - 1) / threads.x,
				(image.height() + threads.y - 1) / threads.y);
	matrixImage<float> *matBlur1 = grayBlur(image, 1, threads, blocks,
											"gpu_gray1.png");
	matrixImage<float> *matBlur2 = grayBlur(image2, 1, threads, blocks,
											"gpu_gray2.png");
	write_image_float(matBlur1, "gpu_blur1.png");
	write_image_float(matBlur2, "gpu_blur2.png");

	lunch_abs_diff(matBlur1, matBlur2, threads, blocks);
	write_image_float(matBlur2, "gpu_abs_diff.png");

	launchMorphOpeningClosing(matBlur2, threads, blocks);
	write_image_float(matBlur2, "gpu_morph.png");

	launchThreshold(matBlur2, threads, blocks);
	write_image_float(matBlur2, "gpu_threshold.png");
	delete matBlur1;
	delete matBlur2;
}
