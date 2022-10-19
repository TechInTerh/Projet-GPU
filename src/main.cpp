#include <cstdio>
#include <iostream>
#include "render_gpu.cuh"
#include "render_cpu.h"
#include <boost/gil/image.hpp>
#include <boost/gil/extension/io/png/read.hpp>
#include <boost/gil/extension/io/png/write.hpp>
#include <boost/gil/extension/io/png.hpp>

#define USE_GPU false
namespace gil = boost::gil;

//using namespace boost::gil;
void writeImage(const std::string &path, gil::rgb8_image_t &image)
{
	write_view(path, gil::view(image), gil::png_tag());
}

gil::rgb8_image_t loadImage(const std::string &path)
{
	gil::rgb8_image_t image;
	read_and_convert_image(path, image, gil::png_tag());
	return image;

}

int main()
{
	gil::rgb8_image_t image = loadImage("img/img_1.png");
	writeImage("img/test.png", image);
#if (USE_GPU)
	use_gpu();
#else
	use_cpu();
#endif
	return 0;
}
