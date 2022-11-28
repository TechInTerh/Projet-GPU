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
	size_t pitch;

	matrixImage(size_t width, size_t height) : width(width), height(height)
	{
		isGPU = false;
		buffer = new T[width * height];
		pitch = width * sizeof(T);
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
		};
		T *gpuBuffer = (T *) cudaMallocPitchX(&pitch, width*sizeof(T), height);
		cudaMemcpy2DX(gpuBuffer, pitch, buffer, width * sizeof(T), width*sizeof(T),
					  height, cudaMemcpyHostToDevice);

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
		cudaMemcpy2DX(cpuBuffer, width * sizeof(T),
					  buffer, pitch,
					  width *sizeof(T),height,cudaMemcpyDeviceToHost);
		cudaFreeX(buffer);
		pitch = width*sizeof(T);
		buffer = cpuBuffer;
		isGPU = false;
	}

	bool isGpu()
	{
		return isGPU;
	}

	matrixImage<T> *deepCopy()
	{
		matrixImage<T> *newMat = new matrixImage<T>(width, height);
		if (isGPU)
		{
			newMat->toGpu();
			cudaMemcpy2DX(newMat->buffer, newMat->pitch, buffer, pitch,
						  width * sizeof(T), height,
						  cudaMemcpyDeviceToDevice);
		}
		else
		{
			memcpy(newMat->buffer, buffer, width * height * sizeof(T));
		}
		return newMat;
	}

	//destructor
	~matrixImage()
	{
		if (isGPU)
		{
			cudaFree(buffer);
		}
		else
		{
			delete[] buffer;
		}
	}
	void abs_diff(matrixImage<T> *mat_diff)
	{
		if (mat_diff->height != height || mat_diff->width != width)
		{
			spdlog::error("substraction can only be used with matrices of same size.");
			return;
		}
		for (size_t w = 0; w < width; w++)
		{
			for (size_t h = 0; h < height; h++)
			{
				T *left_px = this->at(w,h);
				T *right_px = mat_diff->at(w, h);
				T diff = std::abs(*left_px - *right_px);
				this->set(w, h, diff);
			}
		}
	}

};

matrixImage<uchar3> *toMatrixImage(gil::rgb8_image_t &image);

void write_image(matrixImage<uchar3> *matImage, const char *filename);

__device__ __host__
uchar3 createUchar3(unsigned char r, unsigned char g, unsigned char b);
matrixImage<uchar3> * matFloatToMatUchar3(matrixImage<float> * matIn);

#endif //GPGPU_MATRIX_IMAGE_CUH
