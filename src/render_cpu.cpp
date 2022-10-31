#include "render_cpu.hpp"
#include <iostream>
#include <cstddef>
#include <vector>
#include <vector_types.h>
#include <boost/gil/image.hpp>
#include <boost/gil/typedefs.hpp>
#include <boost/gil/image.hpp>
#include <boost/gil/extension/io/png/read.hpp>
#include <boost/gil/extension/io/png/write.hpp>
#include <boost/gil/extension/io/png.hpp>

void _abortError(const char *msg,const char *filename, const char *fname, int line)
{
	spdlog::error("{} ({},file: {}, line: {})", msg, filename,fname, line);
	std::exit(1);
}

template <typename T>
matrixImage<T> * toMatrixImage(gil::rgb8_image_t &image)
{
	gil::rgb8_image_t::const_view_t view = gil::const_view(image);
	assert(view.is_1d_traversable());

	size_t width = view.width();
	size_t height = view.height();
	matrixImage<T> *mat = new matrixImage<T>(width,height);

	for (size_t y = 0; y < height; y+=1)
	{
		auto it = view.row_begin(y);
		for (size_t x = 0; x < width ; x++)
		{

			gil::rgb8_pixel_t pixel = it[x];
			T a = T();
			a.x = gil::at_c<0>(pixel);
			a.y = gil::at_c<1>(pixel);
			a.z = gil::at_c<2>(pixel);
			mat->set(x, y, a);
		}

		// use it[j] to access pixel[i][j]
	}
	return mat;
}
void write_image(matrixImage<uchar3> *matImage)
{
	gil::rgb8_pixel_t *pix = (gil::rgb8_pixel_t *)(matImage->buffer);
	gil::rgb8c_view_t src = gil::interleaved_view(matImage->width, matImage->height, (boost::gil::rgb8_pixel_t const*)(matImage->buffer),matImage->width*
																																		 sizeof(uchar3));
	spdlog::info("Writing into my_file.png");
	gil::write_view("my_file.png",src,gil::png_tag());
}

void toGrayscale(matrixImage<uchar4> *buf_in, matrixImage<float> *buf_out, size_t width, size_t height)
{
	for (size_t w = 0; w < width; w++)
	{
		for (size_t h = 0; h < height ; h++)
		{
			uchar4 *px_in = buf_in->at(h,w);
			float px_out = 0.3 * px_in->x + + 0.59 * px_in->y + 0.11 * px_in->z;
			buf_out->set(h, w, px_out);
		}
	}
}

//  FIXME
//  1. check if operators * and / are defined for uchar4 type and
//	int/float/size_t
//  2. check if types don't cause overflows/undefined/unexpected behaviors

void pxGaussianBlur(
		matrixImage<float> *buf_in,
		matrixImage<float> *buf_out,
		size_t x,
		size_t y,
		const float **kernel,
		const size_t kernel_size,
		const size_t offset)
{
	float px = 0;
	for (size_t k_w = 0; k_w < kernel_size; k_w++)
	{
		for (size_t k_h = 0; k_h < kernel_size; k_w++)
		{
			float *px_tmp = buf_in->at(x + k_w - offset, y + k_w - offset);
			float k_elt = kernel[k_h][k_w];
			px += *px_tmp * k_elt;
		}
	}
	buf_out->set(x, y, px);
}

//  FIXME
//  1. make sure buffer_out is correctly initialized (should be empty)
//  2. ideally find a way to not use 2 buffers
//  3. perhaps optimize looping for compleity reduction on mat_mult
//  4. check if typing of kernel should be changed (to char, float, size_t...)
//  5. Since buffer_in is a grayscale image, maybe the px type shouldn't be uchar4
//  6. init basic kernels (3x3 & 5x5) in .hpp
//  7. find a way not to use kernel_size in arguments


void gaussianBlur(
		matrixImage<float> *buf_in,
		matrixImage<float> *buf_out,
		size_t width,
		size_t height,
		const float **kernel,
		const size_t kernel_size)
{

	size_t offset = kernel_size / 2;
	for (size_t w = offset; w < (width - offset); w++)
	{
		for (size_t h = offset; h < (height - offset); h++)
		{
			pxGaussianBlur(buf_in, buf_out, w, h, kernel, kernel_size, offset);
		}
	}
}
void useCpu(boost::gil::rgb8_image_t &image)
{
	matrixImage<uchar3> *matImg = toMatrixImage<uchar3>(image);
	write_image(matImg);
	delete matImg;
}
