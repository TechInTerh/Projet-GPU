#ifndef GPGPU_GRAY_BLUR_CUH
#define GPGPU_GRAY_BLUR_CUH

#include "matrix_image/matrix_image.cuh"
#include "render_gpu/tools.cuh"

matrixImage<float>* grayBlur(gil::rgb8_image_t& image, size_t numberBlur,
                             dim3 threads, dim3 blocks);
matrixImage<float>* generateKernelGPU(size_t kernel_size);
#endif // GPGPU_GRAY_BLUR_CUH
