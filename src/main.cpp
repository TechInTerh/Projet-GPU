#include <iostream>
#include "render_gpu.cuh"
#include "render_cpu.hpp"
#include <spdlog/spdlog.h>
#include <fstream>

#define USE_GPU false
namespace gil = boost::gil;


gil::rgb8_image_t loadImage(const std::string &path)
{
	spdlog::info("Loading " + path);
	gil::rgb8_image_t image;
	read_and_convert_image(path, image, gil::png_tag());
	return image;

}

int main(int argc, char *argv[])
{
	if (argc < 3)
	{

		spdlog::error("ArgumentError: ./main <file1> <file2>");
		std::exit(1);
	}

	//FIXME add option handling here
	gil::rgb8_image_t image1 = loadImage(argv[1]);
	json bboxes;
	for (int i = 2; i < argc; i++)
	{

		gil::rgb8_image_t image2 = loadImage(argv[i]);
#if (USE_GPU)
		spdlog::info("Using GPU");
		use_gpu(image1, image2);
#else
		spdlog::info("Using CPU");
		useCpu(image1, image2, argv[i], bboxes);
#endif
	}
	std::ofstream of("bounding_boxes");
	of << bboxes;
	return 0;
}
