#include "render_cpu.hpp"
#include <iostream>
#include <cstddef>
#include <vector>
#include <vector_types.h>
#include <algorithm>
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
	spdlog::info("Gaussian Blurring.");
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


float pxDilationErosion(matrixImage<float> *matImg,
		const size_t w,
		const size_t h,
		const size_t se_w,
		const size_t se_h,
		const bool d_or_e)
{
	//FIXME maybe change the way the structuring element is used. Use square
	// centered around current pixel instead of the current pixel being in a corner.

	/*
	 * d_or_e == true => dilation, else erosion.
	 */
	float max_value = 0.f;
	float min_value = 256.f;
	size_t off_w = se_w / 2;
	size_t off_h = se_h / 2;
	for (size_t sw = 0; sw + w - off_w < matImg->width && sw < se_w; sw++)
	{
		if (sw + w < off_w)
			continue;
		for (size_t sh = 0; sh + h - off_h < matImg->height && sh < se_h; sh++)
		{
			if (sh + h < off_h)
				continue;
			max_value = std::max(max_value, *(matImg->at(sw + w - off_w, sh + h - off_h)));
			min_value = std::min(min_value, *(matImg->at(sw + w - off_w, sh + h - off_h)));
		}
	}
	if (d_or_e)
		return max_value;
	return min_value;
}
void dilationErosion(matrixImage<float> *mat_in,
		matrixImage<float> *mat_out,
		const size_t se_w,
		const size_t se_h,
		const bool d_or_e)
{
	/*
	 * d_or_e== true => dilation, else erosion
	 */


	for (size_t w = 0; w < mat_in->width; w++)
	{
		for (size_t h = 0; h < mat_out->height; h++)
		{
			float value = pxDilationErosion(mat_in, w, h, se_w, se_h, d_or_e);
			mat_out->set(w, h, value);
		}
	}
}

//FIXME atm, only rectangle structuring elements with unique pixel center
//	used. check if disks would be better.
void morphOpening(matrixImage<float> *mat_in, matrixImage<float> *mat_out, size_t se_w, size_t se_h)
{
	spdlog::info("Morphological Opening");
	dilationErosion(mat_in, mat_out, se_w, se_h, false);
	dilationErosion(mat_in, mat_out, se_w, se_h, true);
}
void morphClosing(matrixImage<float> *mat_in, matrixImage<float> *mat_out, size_t se_w, size_t se_h)
{
	spdlog::info("Morphological Closing");
	dilationErosion(mat_in, mat_out, se_w, se_h, true);
	dilationErosion(mat_in, mat_out, se_w, se_h, false);
}



//FIXME maybe try to change all operations so we just need an in matrix.
void useCpu(gil::rgb8_image_t &image1, gil::rgb8_image_t &image2)
{
	matrixImage<uchar3> *matImg1 = toMatrixImage(image1);
	matrixImage<float> *matGray1 = new matrixImage<float>(matImg1->width,matImg1->height);
	toGrayscale(matImg1, matGray1);

	matrixImage<uchar3> *matImg2 = toMatrixImage(image2);
	matrixImage<float> *matGray2 = new matrixImage<float>(matImg2->width,matImg2->height);
	toGrayscale(matImg2, matGray2);

	matrixImage<float> *matGBlur1 = new matrixImage<float>(matImg1->width,matImg1->height);
	gaussianBlur(matGray1, matGBlur1);

	matrixImage<float> *matGBlur2 = new matrixImage<float>(matImg2->width,matImg2->height);
	gaussianBlur(matGray2, matGBlur2);

	matrixImage<float> *matGigaBlur_tmp = new matrixImage<float>(matImg1->width,matImg1->height);
	matrixImage<float> *matGigaBlur1 = new matrixImage<float>(matImg1->width,matImg1->height);
	matGigaBlur_tmp->copy(matGBlur1);
	int repeatBlur = 5;
	for (int i = 0; i < repeatBlur; i++)
	{
		gaussianBlur(matGigaBlur_tmp, matGigaBlur1);
		matGigaBlur_tmp->copy(matGigaBlur1);
	}

	matrixImage<float> *matGBlur1_save = new matrixImage<float>(matImg1->width,matImg1->height);
	matGBlur1_save->copy(matGBlur1);

	matGBlur1->abs_diff(matGBlur2);

	matrixImage<float> *matOpening = new matrixImage<float>(matImg1->width,matImg1->height);
	morphOpening(matGBlur1, matOpening, 20, 20);
	matrixImage<float> *matClosing = new matrixImage<float>(matImg1->width,matImg1->height);
	morphClosing(matOpening, matClosing, 50, 50);

	matrixImage<uchar3> *matGray_out = matFloatToMatUchar3(matGray1);
	write_image(matGray_out, "grayscale.png");
	matrixImage<uchar3> *matGBlur_out = matFloatToMatUchar3(matGBlur1_save);
	write_image(matGBlur_out, "gaussian_blur.png");
	matrixImage<uchar3> *matGigaBlur_out = matFloatToMatUchar3(matGigaBlur1);
	write_image(matGigaBlur_out, "giga_blur.png");

	delete matImg1;
	delete matImg2;
	delete matGray1;
	delete matGray2;
	delete matGray_out;
	delete matGigaBlur_tmp;
	delete matGBlur1;
	delete matGBlur2;
	delete matGBlur1_save;
	delete matGBlur_out;
	delete matGigaBlur1;
	delete matGigaBlur_out;
	delete matOpening;
	delete matClosing;
}
