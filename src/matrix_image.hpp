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
	void sub(matrixImage<T> *mat2)
	{
		for(size_t h = 0; h < height; w++)
		{
			for(size_t w = 0; w < width; w++)
			{
				*this->at(w, h) = this->at(w, h) - mat->at(w, h);
			}
		}
	}
};

matrixImage<uchar4> *toMatrixImage(gil::rgb8_image_t &image);

#endif //GPGPU_RENDER_CPU_HPP
