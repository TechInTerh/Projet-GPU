#include <cstdio>
#include <iostream>
#include "render_gpu.cuh"
#include "render_cpu.h"
#include <boost/gil/image.hpp>
#include <boost/gil/extension/io/png/read.hpp>
#include <boost/gil/extension/io/png/write.hpp>
#include <boost/gil/extension/io/png.hpp>

#define USE_GPU false
using namespace boost::gil;
int main()
{
#if (USE_GPU)
	use_gpu();
#else
	use_cpu();
#endif
	rgb8_image_t input;
	std::string image_path = "img/img.png";


	read_and_convert_image(image_path,input, png_tag());
	/*
	std::cout << input.height() << "\n";*/
	write_view("img/test.png", view(input),png_tag());
	return 0;
}
