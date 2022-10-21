#include "render_cpu.hpp"
#include <iostream>
#include <cstddef>
#include <vector>
#include <vector_types.h>

uchar4 * list_uchar4(gil::rgb8_image_t &image)
{
	gil::rgb8_image_t::const_view_t view = gil::const_view(image);
	assert(view.is_1d_traversable());

	size_t width = view.width();
	size_t height = view.height();
	uchar4 *buf = new uchar4[width * height];

	for (size_t i = 0; i < height; i+=1)
	{
		auto it = view.row_begin(i);
		for (size_t j = 0; j < width ; j++)
		{

			gil::rgb8_pixel_t pixel = it[j];
			buf[i*width+j].x = gil::at_c<0>(pixel);
			buf[i*width+j].y = gil::at_c<1>(pixel);
			buf[i*width+j].z = gil::at_c<2>(pixel);
			buf[i*width+j].w = 0;
		}

		// use it[j] to access pixel[i][j]
	}
	return buf;
}

void useCpu(boost::gil::rgb8_image_t &image)
{
	uchar4 * buffer = list_uchar4(image);
	delete buffer;
}
