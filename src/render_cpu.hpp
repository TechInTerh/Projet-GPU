#ifndef GPGPU_RENDER_CPU_HPP
#define GPGPU_RENDER_CPU_HPP

#include <boost/gil/typedefs.hpp>
#include <boost/gil/image.hpp>
#include <boost/gil/extension/io/png/read.hpp>
#include <boost/gil/extension/io/png/write.hpp>
#include <boost/gil/extension/io/png.hpp>
#include <vector_types.h>

namespace gil = boost::gil;

void useCpu(gil::rgb8_image_t &image);
uchar4 * list_uchar4(gil::rgb8_image_t &image);

#endif //GPGPU_RENDER_CPU_HPP
