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



matrixImage<uchar3> * matFloatToMatUchar3(matrixImage<float> * matIn)
{
	spdlog::info("Converting to uchar3");
	matrixImage<uchar3> *matOut = new matrixImage<uchar3>(matIn->width,matIn->height);
	for (size_t w = 0; w < matIn->width; w++)
	{
		for (size_t h = 0; h < matIn->height; h++)
		{
			uchar3 val = uchar3();
			val.x = ceil(*matIn->at(w,h));
			val.y = val.x;
			val.z = val.x;
			matOut->set(w,h, val);
		}
	}
	return matOut;
}
void toGrayscale(matrixImage<uchar3> *buf_in, matrixImage<float> *buf_out,
				 size_t width, size_t height)
{
	spdlog::info("To Grayscale");
	for (size_t w = 0; w < width; w++)
	{
		for (size_t h = 0; h < height; h++)
		{
			uchar3 *px_in = buf_in->at(w, h);
			float px_out = 0.3 * px_in->x + +0.59 * px_in->y + 0.11 * px_in->z;
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
	matrixImage<uchar3> *matImg = toMatrixImage(image);
	matrixImage<float> *matGray = new matrixImage<float>(matImg->width,matImg->height);
	toGrayscale(matImg,matGray,matImg->width,matImg->height);
	matrixImage<uchar3> *matGray2 = matFloatToMatUchar3(matGray);
	write_image(matGray2);
	delete matGray;
	delete matGray2;
	delete matImg;
}
