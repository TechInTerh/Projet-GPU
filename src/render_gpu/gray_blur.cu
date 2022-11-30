#include "gray_blur.cuh"
#include "tools.cuh"

__global__ void
grayscale(uchar3 *matImg, float *matOut, size_t width, size_t height,
		  size_t pitch_in, size_t pitch_out)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t idy = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx >= width || idy >= height)
	{
		return;
	}
	uchar3 *px_in = eltPtrUchar3(matImg, idx, idy, pitch_in);
	float val_out = 0.3 * px_in->x + 0.59 * px_in->y + 0.11 * px_in->z;
	float *px_out = eltPtrFloat(matOut, idx, idy, pitch_out);
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
	if (idx >= width || idy >= height)
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
				float *px_in = eltPtrFloat(matIn, idx + k_w - offset,
												  idy + k_h - offset, pitch_in);
				val_out +=
						*px_in * (*eltPtrFloat(kernel, k_w, k_h, kernel_pitch));
			}
		}
	}
	float *px_out = eltPtrFloat(matOut, idx, idy, pitch_out);
	*px_out = my_min(val_out, 255.f);

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

matrixImage<float> *
grayBlur(gil::rgb8_image_t &image, size_t numberBlur, dim3 threads,
		 dim3 blocks)
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

