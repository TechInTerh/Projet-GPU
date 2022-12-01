#include <fstream>
#include <iostream>
#include <spdlog/spdlog.h>

#include "render_cpu.hpp"
#include "render_gpu/render_gpu.cuh"

namespace gil = boost::gil;

gil::rgb8_image_t loadImage(const std::string& path)
{
    spdlog::info("Loading " + path);
    gil::rgb8_image_t image;
    read_and_convert_image(path, image, gil::png_tag());
    return image;
}

int main(int argc, const char* argv[])
{
    bool isUseGpu = false;
    if (argc < 3)
    {
        spdlog::error(
            "ArgumentError: ./main [--gpu] <file1> <file2> [file3...]");
        std::exit(1);
    }
    std::string first_image;
    int i;
    if (argv[1] == std::string("--gpu"))
    {
        first_image = argv[2];
        isUseGpu = true;
        i = 3;
    }
    else
    {
        first_image = argv[1];
        i = 2;
    }
    gil::rgb8_image_t image1 = loadImage(first_image);
    json bboxes;
    for (; i < argc; i++)
    {
        gil::rgb8_image_t image2 = loadImage(argv[i]);
        if (isUseGpu)
        {
            spdlog::info("Using GPU");
            useGpu(image1, image2, argv[i], bboxes);
        }
        else
        {
            spdlog::info("Using CPU");
            useCpu(image1, image2, argv[i], bboxes);
        }
    }
    std::ofstream of("bounding_boxes.json");
    of << bboxes;
    if (isUseGpu)
    {
        cudaDeviceReset();
    }
    return 0;
}
