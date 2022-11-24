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




void useCpu(gil::rgb8_image_t &image);


void toGrayscale(matrixImage<uchar3> *buf_in, matrixImage<float> *buf_out,
				 size_t width, size_t height);

#endif //GPGPU_RENDER_CPU_HPP
