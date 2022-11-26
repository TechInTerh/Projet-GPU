#ifndef GPGPU_RENDER_CPU_HPP
#define GPGPU_RENDER_CPU_HPP

#include <boost/gil/typedefs.hpp>
#include <boost/gil/image.hpp>
#include <boost/gil/extension/io/png/read.hpp>
#include <boost/gil/extension/io/png/write.hpp>
#include <boost/gil/extension/io/png.hpp>
#include <vector_types.h>
#include <spdlog/spdlog.h>
#include "matrix_image/matrix_image.h"



void useCpu(gil::rgb8_image_t &image1, gil::rgb8_image_t &image2);


void toGrayscale(matrixImage<uchar4> *buf_in, matrixImage<float> *buf_out);
void gaussianBlur(matrixImage<float> *buf_in, matrixImage<float> *buf_out);

#endif //GPGPU_RENDER_CPU_HPP
