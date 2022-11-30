#include "tools.cuh"

__device__ float my_max(float a, float b)
{
	return a > b ? a : b;
}

__device__ float my_min(float a, float b)
{
	return a < b ? a : b;
}


__device__ __host__
float* eltPtrFloat(float *baseAddress, size_t col, size_t row, size_t pitch)
{
	return (float *) ((char *) baseAddress + row * pitch +
				  col * sizeof(float));
}
__device__ __host__
uchar3* eltPtrUchar3(uchar3 *baseAddress, size_t col, size_t row, size_t pitch)
{
	return (uchar3 *) ((char *) baseAddress + row * pitch +
				  col * sizeof(uchar3));
}
void writeImageFloat(matrixImage<float> *mat, std::string path)
{
	mat->toCpu();
	matrixImage<uchar3> *tmp = matFloatToMatUchar3(mat);
	write_image(tmp, path.c_str());
	delete tmp;
	mat->toGpu();

}
