#ifndef GPGPU_RENDER_GPU_CUH
#define GPGPU_RENDER_GPU_CUH

#include "env.cuh"
#include "wrapper_json.hpp"

void useGpu(gil::rgb8_image_t& image, gil::rgb8_image_t& image2,
            const char* filename, json& bboxes);

#endif // GPGPU_RENDER_GPU_CUH
