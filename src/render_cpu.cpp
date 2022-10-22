#include "render_cpu.hpp"
#include <iostream>
#include <cstddef>
#include <vector>
#include <vector_types.h>


matrixImage * toMatrixImage(gil::rgb8_image_t &image)
{
	gil::rgb8_image_t::const_view_t view = gil::const_view(image);
	assert(view.is_1d_traversable());

	size_t width = view.width();
	size_t height = view.height();
	matrixImage* mat = new matrixImage(width,height);

	for (size_t i = 0; i < height; i+=1)
	{
		auto it = view.row_begin(i);
		for (size_t j = 0; j < width ; j++)
		{

			gil::rgb8_pixel_t pixel = it[j];
			uchar4 a = uchar4();
			a.x = gil::at_c<0>(pixel);
			a.y = gil::at_c<1>(pixel);
			a.z = gil::at_c<2>(pixel);
			a.w = 0;
			mat->set(i, j, a);
		}

		// use it[j] to access pixel[i][j]
	}
	return mat;
}

void useCpu(boost::gil::rgb8_image_t &image)
{
	matrixImage * matImg = toMatrixImage(image);

	delete matImg;
}

void toGrayscale(matrixImage *buffer, size_t width, size_t height)
{
	for (size_t w = 0; w < width; w++)
	{
		for (size_t h = 0; h < height ; h++)
		{
			uchar4 *pixel = buffer->at(h,w);
			
		}
	}
}
