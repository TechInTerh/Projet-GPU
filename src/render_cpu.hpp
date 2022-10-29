#ifndef GPGPU_RENDER_CPU_HPP
#define GPGPU_RENDER_CPU_HPP

#include <boost/gil/typedefs.hpp>
#include <boost/gil/image.hpp>
#include <boost/gil/extension/io/png/read.hpp>
#include <boost/gil/extension/io/png/write.hpp>
#include <boost/gil/extension/io/png.hpp>
#include <vector_types.h>
#include <spdlog/spdlog.h>

namespace gil = boost::gil;


void _abortError(const char *msg, const char *filename, const char *fname, int line);

#define abortError(msg) _abortError(msg,__FILE__, __FUNCTION__, __LINE__)

struct kernel
{
	float **buf;
	size_t size;
};

template<typename T>
struct matrixImage
{
	T *buffer;
	size_t width;
	size_t height;

	matrixImage(size_t width, size_t height) : width(width), height(height)
	{
		buffer = new T[width * height];
	}

	T *at(size_t x, size_t y) const
	{
		if (x > width || y > height)
		{
			abortError("Access out of bound");
		}
		return &buffer[y * width + x];
	}

	void set(size_t x, size_t y, T &value)
	{
		*this->at(x, y) = value;
	}
};


template<typename T>
matrixImage<T> *toMatrixImage(gil::rgb8_image_t &image);

void useCpu(gil::rgb8_image_t &image);

matrixImage<uchar4> *toMatrixImage(gil::rgb8_image_t &image);

void toGrayscale(matrixImage<uchar4> *buf_in, matrixImage<float> *buf_out);
void gaussianBlur(matrixImage<float> *buf_in, matrixImage<float> *buf_out);

#endif //GPGPU_RENDER_CPU_HPP
