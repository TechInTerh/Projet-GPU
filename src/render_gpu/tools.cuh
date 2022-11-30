#ifndef GPGPU_TOOLS_CUH
#define GPGPU_TOOLS_CUH

#include "matrix_image/matrix_image.cuh"
__device__ float my_max(float a, float b);

__device__ float my_min(float a, float b);

__device__ __host__
float * eltPtrFloat(float *baseAddress, size_t col, size_t row, size_t pitch);
__device__ __host__
uchar3* eltPtrUchar3(uchar3 *baseAddress, size_t col, size_t row, size_t pitch);

void writeImageFloat(matrixImage<float> *mat, std::string path);
#endif //GPGPU_TOOLS_CUH
