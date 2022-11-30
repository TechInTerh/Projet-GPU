#include "render_gpu.cuh"

#include "tools.cuh"
#include "gray_blur.cuh"
#include <iostream>
#include <map>
#include "cuda_runtime_api.h"
#include "bounding_box.cuh"


__global__ void
absDiff(float *matIn, float *matOut, size_t width, size_t height,
		size_t pitch)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx >= width || idy >= height)
	{
		return;
	}
	float *px_in = eltPtrFloat(matIn, idx, idy, pitch);
	float *px_out = eltPtrFloat(matOut, idx, idy, pitch);
	*px_out = std::abs(*px_in - *px_out);
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
	absDiff<<<blocks, threads>>>(matBlur1->buffer, matBlur2->buffer,
								 matBlur2->width, matBlur2->height,
								 matBlur2->pitch);
	cudaDeviceSynchronizeX();
}


__global__ void dilatationErosion(float *matIn, float *matOut, size_t width,
								  size_t height, size_t pitch_in,
								  size_t pitch_out, size_t se_w,
								  size_t se_h, bool d_or_e)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx >= width || idy >= height)
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
				float *px_in = eltPtrFloat(matIn, idx + sw - off_w,
											 idy + sh - off_h, pitch_in);
				if (d_or_e)
					max_value = my_max(max_value, *px_in);

				else
					min_value = my_min(min_value, *px_in);

			}
		}
	}
	float *px_out = eltPtrFloat(matOut, idx, idy, pitch_out);
	if (d_or_e)
		*px_out = max_value;
	else
		*px_out = min_value;
}

__global__ void generateHistogram(float *matIn, size_t width, size_t height,
								  size_t pitch, int *histogram,
								  size_t hist_size)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx >= width || idy >= height)
	{
		return;
	}
	float *px_in = eltPtrFloat(matIn, idx, idy, pitch);
	int value = (int) floor(*px_in);
	atomicAdd(&histogram[value], 1);
}

int findMeanIntensity(int *histo, size_t nb_px)
{
	float sigmas[256] = {0.f};
	for (int i = 1; i < 256; i++)
	{
		float wb, wf, mu_b, mu_f, count_b;
		wb = wf = mu_b = mu_f = count_b = 0.f;
		for (int j = 0; j < i; j++)
		{
			count_b += histo[j];
			mu_b += histo[j] * j;
		}
		wb = count_b / (float) nb_px;
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
	if (idx >= width || idy >= height)
	{
		return;
	}
	float *px_in = eltPtrFloat(matIn, idx, idy, pitch);
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
	cudaMemcpyX(gpu_histo, histo, size_histo * sizeof(int),
				cudaMemcpyHostToDevice);

	generateHistogram<<<blocks, threads>>>(matIn->buffer, matIn->width,
										   matIn->height, matIn->pitch,
										   gpu_histo,
										   size_histo);
	cudaDeviceSynchronizeX();
	cudaMemcpyX(histo, gpu_histo, size_histo * sizeof(int),
				cudaMemcpyDeviceToHost);

	int mean = findMeanIntensity(histo, matIn->width * matIn->height);

	thresholding<<<blocks, threads>>>(matIn->buffer, matIn->width,
									  matIn->height,
									  matIn->pitch, mean);
	cudaDeviceSynchronizeX();
	cudaFreeX(gpu_histo);
}

__global__ void
setOnIndex(float *matIn, float *matOut, size_t width, size_t height,
		   size_t pitch_in, size_t pitch_out, int* index)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx >= width || idy >= height)
	{
		return;
	}
	float *px_in = eltPtrFloat(matIn, idx, idy, pitch_in);
	float *px_out = eltPtrFloat(matOut, idx, idy, pitch_out);
	if (*px_in == 255.f)
		*px_out = (float)atomicAdd(index, 1);
	else
		*px_out = 0.f;
}

__global__ void
labelNeighbors(float *matIn, float *matOut, size_t width, size_t height,
			   size_t pitch_in, size_t pitch_out, int *isChanged)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx >= width || idy >= height)
	{
		return;
	}
	float *px_in_tmp = eltPtrFloat(matIn, idx, idy, pitch_in);
	if (*px_in_tmp == 0.f)
		return;
	float min_value = *px_in_tmp;
	for (size_t sw = 0; sw <= 2; sw++)
	{
		for (size_t sh = 0; sh <= 2; sh++)
		{
			if (idx + sw - 1 < width && idy + sh - 1 < height)
			{
				float *px_in = eltPtrFloat(matIn, idx + sw - 1,
											 idy + sh - 1, pitch_in);
				if (*px_in != 0.f)
				{
					min_value = my_min(min_value, *px_in);
				}
			}
		}
	}
	float *px_out = eltPtrFloat(matOut, idx, idy, pitch_out);
	if (min_value != *px_out)
	{
		atomicAdd(isChanged, 1);
	}
	*px_out = min_value;
}

void launchMorphOpeningClosing(matrixImage<float> *matIn, dim3 threads,
							   dim3 blocks)
{
	size_t size1_w = 0.02 * matIn->width;
	size_t size1_h = 0.02 * matIn->height;
	size_t size2_w = 0.05 * matIn->width;
	size_t size2_h = 0.05 * matIn->height;

	spdlog::info("Lunching morph closing");
	matrixImage<float> *matOut = matIn->deepCopy();
	dilatationErosion<<<blocks, threads>>>(matIn->buffer, matOut->buffer,
										   matIn->width, matIn->height,
										   matIn->pitch, matOut->pitch, size1_w,
										   size1_h,
										   true);
	cudaDeviceSynchronizeX();
	dilatationErosion<<<blocks, threads>>>(matOut->buffer, matIn->buffer,
										   matIn->width, matIn->height,
										   matIn->pitch, matOut->pitch, size1_w,
										   size1_h,
										   false);
	cudaDeviceSynchronizeX();
	spdlog::info("Lunching morph opening");
	dilatationErosion<<<blocks, threads>>>(matIn->buffer, matOut->buffer,
										   matIn->width, matIn->height,
										   matIn->pitch, matOut->pitch, size2_w,
										   size2_h,
										   false);

	cudaDeviceSynchronizeX();
	dilatationErosion<<<blocks, threads>>>(matOut->buffer, matIn->buffer,
										   matIn->width, matIn->height,
										   matIn->pitch, matOut->pitch, size2_w,
										   size2_h,
										   true);
	cudaDeviceSynchronizeX();
	delete matOut;
}

matrixImage<float> *
launchLabelisation(matrixImage<float> *matIn, dim3 threads, dim3 blocks)
{
	spdlog::info("Lunching labelisation");
	matrixImage<float> *matOutIndex = matIn->deepCopy();
	int one = 1;
	int *index = (int *)cudaMallocX(sizeof(int));
	cudaMemcpyX(index, &one, sizeof(int), cudaMemcpyHostToDevice);
	setOnIndex<<<blocks, threads>>>(matIn->buffer, matOutIndex->buffer,
									matIn->width, matIn->height,
									matIn->pitch, matOutIndex->pitch, index);
	cudaDeviceSynchronizeX();
	int *isChanged_gpu = (int *) cudaMallocX(sizeof(int));
	int isChanges = 1;
	matrixImage<float> *matOutLabel = matOutIndex->deepCopy();
	matrixImage<float> *ret = matOutLabel;
	while (isChanges >= 1)
	{
		isChanges = 0;
		cudaMemcpyX(isChanged_gpu, &isChanges, sizeof(int),
					cudaMemcpyHostToDevice);

		labelNeighbors<<<blocks, threads>>>(matOutIndex->buffer, matOutLabel->buffer,
											matOutLabel->width, matOutLabel->height,
											matOutIndex->pitch, matOutLabel->pitch,
											isChanged_gpu);
		cudaDeviceSynchronizeX();
		cudaMemcpyX(&isChanges, isChanged_gpu, sizeof(int),
					cudaMemcpyDeviceToHost);
		ret = matOutLabel;
		matOutLabel = matOutIndex;
		matOutIndex = ret;
	}

	cudaFreeX(isChanged_gpu);
	delete matOutLabel;
	return ret;
}


__global__ void
multiplyValue(float *matIn, size_t width, size_t height, size_t pitch_in,
			  float value)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx >= width || idy >= height)
	{
		return;
	}
	float *px_in = eltPtrFloat(matIn, idx, idy, pitch_in);
	*px_in *= value;
}


void useGpu(gil::rgb8_image_t &image, gil::rgb8_image_t &image2,
			const char *filename,json &bboxes)
{
	dim3 threads(32, 32);
	dim3 blocks((image.width() + threads.x - 1) / threads.x,
				(image.height() + threads.y - 1) / threads.y);
	matrixImage<float> *matBlur1 = grayBlur(image, 1, threads, blocks);
	matrixImage<float> *matBlur2 = grayBlur(image2, 1, threads, blocks);

	lunch_abs_diff(matBlur1, matBlur2, threads, blocks);

	launchMorphOpeningClosing(matBlur2, threads, blocks);

	launchThreshold(matBlur2, threads, blocks);

	matrixImage<float> *matLabel = launchLabelisation(matBlur2, threads, blocks);
	matLabel->toCpu();

	std::vector<std::vector<size_t>> boundingboxes;
	get_bounding_boxes(matLabel, boundingboxes);
	for (int i = 0; i < 2; i++)
	{
		std::cout << "[" << boundingboxes[i][0] << ", " << boundingboxes[i][1] << ", " << boundingboxes[i][2] << ", " << boundingboxes[i][3] << "]"<< std::endl;
	}

	std::string f(filename);
	auto base_filename = f.substr(f.find_last_of("/") + 1);
	bboxes[base_filename] = boundingboxes;

	delete matBlur1;
	delete matBlur2;
	delete matLabel;

}
