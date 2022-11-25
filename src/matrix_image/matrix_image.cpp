#include "matrix_image.h"

void write_image(matrixImage<uchar3> *matImage, const char *filename)
{
	gil::rgb8c_view_t src = gil::interleaved_view(matImage->width,
												  matImage->height,
												  (boost::gil::rgb8_pixel_t const *) (matImage->buffer),
												  matImage->width *
												  sizeof(uchar3));
	spdlog::info("Writing into {}", filename);
	gil::write_view(filename, src, gil::png_tag());
}

matrixImage<uchar3> *toMatrixImage(gil::rgb8_image_t &image)
{
	gil::rgb8_image_t::const_view_t view = gil::const_view(image);
	assert(view.is_1d_traversable());

	size_t width = view.width();
	size_t height = view.height();
	matrixImage<uchar3> *mat = new matrixImage<uchar3>(width, height);

	for (size_t y = 0; y < height; y += 1)
	{
		auto it = view.row_begin(y);
		for (size_t x = 0; x < width; x++)
		{

			gil::rgb8_pixel_t pixel = it[x];
			uchar3 a = uchar3();
			a.x = gil::at_c<0>(pixel);
			a.y = gil::at_c<1>(pixel);
			a.z = gil::at_c<2>(pixel);
			mat->set(x, y, a);
		}

		// use it[j] to access pixel[i][j]
	}
	return mat;
}
