#ifndef GPGPU_MATRIX_IMAGE_CUH
#define GPGPU_MATRIX_IMAGE_CUH

#include <vector_types.h>
#include <cuda_runtime_api.h>
#include "env.cuh"
#include "cudaFunctions/cudaFunctions.cuh"

template<typename T>
struct matrixImage
{
	T *buffer;
	size_t width;
	size_t height;
	bool isGPU;

	matrixImage(size_t width, size_t height) : width(width), height(height)
	{
		isGPU = false;
		buffer = new T[width * height];
	}

	__device__ __host__ //Avaible in both CPU and GPU.
	T *at(size_t x, size_t y)
	{
		return &buffer[y * width + x];
	}
	__device__ __host__
	void set(size_t w, size_t h, T value)
	{
		*this->at(w, h) = value;
	}

	//create a version the GPU can use.

	void toGpu()
	{
		if (isGPU)
		{
			return;
		}
		T *gpuBuffer;
		cudaMalloc(&gpuBuffer, width * height * sizeof(T));
		cudaMemcpy(gpuBuffer, buffer, width * height * sizeof(T),
				   cudaMemcpyHostToDevice);

		delete[] buffer;

		buffer = gpuBuffer;
		isGPU = true;
	}

	void toCpu()
	{
		if (!isGPU)
		{
			return;
		}
		T *cpuBuffer = new T[width * height];
		cudaMemcpy(cpuBuffer, buffer, width * height * sizeof(T),
				   cudaMemcpyDeviceToHost);
		cudaFree(buffer);
		buffer = cpuBuffer;
		isGPU = false;
	}

	bool isGpu()
	{
		return isGPU;
	}
};

matrixImage<uchar3> *toMatrixImage(gil::rgb8_image_t &image);

void write_image(matrixImage<uchar3> *matImage);

__device__ __host__
uchar3 createUchar3(unsigned char r, unsigned char g, unsigned char b);
#endif //GPGPU_MATRIX_IMAGE_CUH
