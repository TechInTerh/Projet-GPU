#include "render_cpu.hpp"
#include <iostream>
#include <cstddef>
#include <vector>
#include <vector_types.h>

void _abortError(const char *msg,const char *filename, const char *fname, int line)
{
	spdlog::error("{} ({},file: {}, line: {})", msg, filename,fname, line);
	std::exit(1);
}

template <typename T>
matrixImage<T> * toMatrixImage(gil::rgb8_image_t &image)
{
	std::cout << "in toMatrixImage.\n";
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
			a.w = 0;
			mat->set(x, y, a);
		}

		// use it[j] to access pixel[i][j]
	}
	return mat;
}

void useCpu(boost::gil::rgb8_image_t &image)
{
	matrixImage<uchar4> *matImg = toMatrixImage<uchar4>(image);

	matrixImage<float> *grayImg = new matrixImage<float>(matImg->width, matImg->height);
	toGrayscale(matImg, grayImg);

	matrixImage<float> *gaussBlurImg = new matrixImage<float>(matImg->width, matImg->height);
	gaussianBlur(grayImg, gaussBlurImg);

	delete matImg;
	delete grayImg;
	delete gaussBlurImg;
}
void toGrayscale(matrixImage<uchar4> *buf_in, matrixImage<float> *buf_out)
{
	size_t width = buf_in->width;
	size_t height = buf_in->height;
	// FIXME add assert in case buf_in size != buf_out size
	for (size_t w = 0; w < width; w++)
	{
		for (size_t h = 0; h < height ; h++)
		{
			uchar4 *px_in = buf_in->at(w,h);
			float px_out = 0.3 * px_in->x + + 0.59 * px_in->y + 0.11 * px_in->z;
			buf_out->set(w, h, px_out);
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
		const size_t offset)
{
	float px = 0;
	const size_t kernel_size = 3;
	const float ker[kernel_size][kernel_size] = {{0.0625, 0.125, 0.0625},{0.125, 0.25, 0.125},{0.0625, 0.125, 0.0625}};
	for (size_t k_w = 0; k_w < kernel_size; k_w++)
	{
		for (size_t k_h = 0; k_h < kernel_size; k_h++)
		{
			float *px_tmp = buf_in->at(x + k_w - offset, y + k_h - offset);
			float k_elt = ker[k_h][k_w];
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
//  8. maybe add padding to image so kernel passes on all px of input image


void gaussianBlur(
		matrixImage<float> *buf_in,
		matrixImage<float> *buf_out)
{
	const size_t kernel_size = 3; //FIXME change kernel and kernel size to a
				      //struct an define a kernel generator
				      //function of its size
	size_t offset = kernel_size / 2;
	size_t width = buf_in->width;
	size_t height = buf_in->height;
	//FIXME add assert to check buf_in size == buf_out size
	for (size_t w = offset; w < (width - offset); w++)
	{
		for (size_t h = offset; h < (height - offset); h++)
		{
			pxGaussianBlur(buf_in, buf_out, w, h, offset);
		}
	}
}
