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
[[gnu::noinline]]
void _abortError(const char *msg, const char *fname, int line)
{
	spdlog::error("{} ({}, line: {})", msg, fname, line);
	std::exit(1);
}

#define abortError(msg) _abortError(msg, __FUNCTION__, __LINE__)

struct matrixImage
{
	uchar4* buffer;
	size_t width;
	size_t height;

	matrixImage(size_t width, size_t height) : width(width), height(height)
	{
		buffer = new uchar4[width*height];
	}

	uchar4* at(size_t x,size_t y) const
	{
		if (x>width || y > height)
		{
			abortError("Access out of bound");
		}
		return &buffer[y*width+x];
	}
	void set(size_t x, size_t y, uchar4 &value)
	{
		*this->at(x,y) = value;
	}
};
void useCpu(gil::rgb8_image_t &image);
matrixImage * toMatrixImage(gil::rgb8_image_t &image);
void toGrayscale(matrixImage *buffer, size_t width, size_t height);

#endif //GPGPU_RENDER_CPU_HPP
