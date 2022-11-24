#include <iostream>
#include "render_gpu.cuh"
#include "render_cpu.hpp"
#include <spdlog/spdlog.h>
#define USE_GPU true
namespace gil = boost::gil;

//using namespace boost::gil;
void writeImage(const std::string &path, gil::rgb8_image_t &image)
{
	spdlog::info("Writing " + path);
	write_view(path, gil::view(image), gil::png_tag());
}

gil::rgb8_image_t loadImage(const std::string &path)
{
	spdlog::info("Loading " + path);
	gil::rgb8_image_t image;
	read_and_convert_image(path, image, gil::png_tag());
	return image;

}

int main()
{

	gil::rgb8_image_t image = loadImage("img/img_1.png");
	writeImage("img/test.png", image);
#if (USE_GPU)
	spdlog::info("Using GPU");
	use_gpu();
#else
	spdlog::info("Using CPU");
	useCpu(image);
#endif
	return 0;
}
