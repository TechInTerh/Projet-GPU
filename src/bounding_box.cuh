#ifndef GPGPU_BOUNDING_BOX_CUH
#define GPGPU_BOUNDING_BOX_CUH

#include <cmath>
#include <string>
#include <map>
#include "matrix_image/matrix_image.cuh"

bool map_contains(std::map<float, int> map, float key);

void get_bounding_boxes(matrixImage<float> *mat_in,
						std::vector<std::vector<size_t>> &boundingboxes);

void new_bounding_box(std::vector<std::vector<size_t>> &boundingboxes, size_t w,
					  size_t h);

#endif //GPGPU_BOUNDING_BOX_CUH
