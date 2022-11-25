#include <iostream>
#include "render_gpu.cuh"
#include "render_cpu.hpp"
#include <spdlog/spdlog.h>
#define USE_GPU false
namespace gil = boost::gil;


gil::rgb8_image_t loadImage(const std::string &path)
{
	spdlog::info("Loading " + path);
	gil::rgb8_image_t image;
	read_and_convert_image(path, image, gil::png_tag());
	return image;

}

int main(int argc, const char *argv[])
{
	//FIXME add option handling here
	gil::rgb8_image_t image = loadImage("../img/img_1.png");
#if (USE_GPU)
	spdlog::info("Using GPU");
	use_gpu();
#else
	spdlog::info("Using CPU");
	useCpu(image);
#endif
	return 0;
}
