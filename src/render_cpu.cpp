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

void
_abortError(const char *msg, const char *filename, const char *fname, int line)
{
	spdlog::error("{} ({},file: {}, line: {})", msg, filename, fname, line);
	std::exit(1);
}


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
void toGrayscale(matrixImage<uchar3> *buf_in, matrixImage<float> *buf_out)
{
	spdlog::info("To Grayscale");
	size_t width = buf_in->width;
	size_t height = buf_in->height;
	// FIXME add assert in case buf_in size != buf_out size
	for (size_t w = 0; w < width; w++)
	{
		for (size_t h = 0; h < height; h++)
		{
			uchar3 *px_in = buf_in->at(w, h);
			float px_out = 0.3 * px_in->x + 0.59 * px_in->y + 0.11 * px_in->z;
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

void useCpu(boost::gil::rgb8_image_t &image)
{
	matrixImage<uchar3> *matImg = toMatrixImage(image);
	matrixImage<float> *matGray = new matrixImage<float>(matImg->width,matImg->height);
	toGrayscale(matImg, matGray);

	matrixImage<float> *matGBlur = new matrixImage<float>(matImg->width,matImg->height);
	gaussianBlur(matGray, matGBlur);
	
	matrixImage<float> *matGigaBlur_in = new matrixImage<float>(matImg->width,matImg->height);
	matrixImage<float> *matGigaBlur_out = new matrixImage<float>(matImg->width,matImg->height);
	matGigaBlur_in->copy(matGBlur);
	int repeatBlur = 5;
	for (int i = 0; i < repeatBlur; i++)
	{
		gaussianBlur(matGigaBlur_in, matGigaBlur_out);
		matGigaBlur_in->copy(matGigaBlur_out);
	}
	matrixImage<uchar3> *matGray2 = matFloatToMatUchar3(matGray);
	write_image(matGray2, "grayscale.png");
	matrixImage<uchar3> *matGBlur2 = matFloatToMatUchar3(matGBlur);
	write_image(matGBlur2, "gaussian_blur.png");
	matrixImage<uchar3> *matGigaBlur2 = matFloatToMatUchar3(matGigaBlur_out);
	write_image(matGigaBlur2, "giga_blur.png");
	delete matGray;
	delete matGray2;
	delete matImg;
	delete matGigaBlur_in;
	delete matGigaBlur_out;

}
