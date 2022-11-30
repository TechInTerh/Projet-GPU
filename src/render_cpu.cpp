#include "render_cpu.hpp"
#include <iostream>
#include <cstddef>
#include <vector>
#include <vector_types.h>
#include <queue>
#include <algorithm>
#include <math.h>
#include <boost/gil/typedefs.hpp>
#include <cmath>
#include <string>

void
_abortError(const char *msg, const char *filename, const char *fname, int line)
{
	spdlog::error("{} ({},file: {}, line: {})", msg, filename, fname, line);
	std::exit(1);
}


void toGrayscale(matrixImage<uchar3> *buf_in, matrixImage<float> *buf_out)
{
	spdlog::info("To Grayscale");
	size_t width = buf_in->width;
	size_t height = buf_in->height;
	for (size_t w = 0; w < width; w++)
	{
		for (size_t h = 0; h < height; h++)
		{
			uchar3 *px_in = buf_in->at(w, h);
			float px_out = 0.3 * px_in->x + 0.59 * px_in->y + 0.11 * px_in->z;
			buf_out->set(w, h, px_out);
		}
	}
}


float **generate_kernel(float **kernel, size_t ker_size)
{
	float sigma = 1;
	float mean = ker_size / 2;
	float sum = 0;
	for (size_t x = 0; x < ker_size; ++x)
	{
		for (size_t y = 0; y < ker_size; ++y)
		{
			kernel[x][y] = std::exp(-0.5 * (std::pow((x - mean) / sigma, 2.0) +
						std::pow((y - mean) / sigma,
							2.0))) /
				(2 * M_PI * sigma * sigma);
			// Accumulate the kernel values
			sum += kernel[x][y];
		}
	}

	// Normalize the kernel
	for (size_t x = 0; x < ker_size; ++x)
		for (size_t y = 0; y < ker_size; ++y)
			kernel[x][y] /= sum;
	return kernel;
}

//  FIXME
//  1. check if operators * and / are defined for uchar4 type and
//	int/float/size_t
//  2. check if types don't cause overflows/undefined/unexpected behaviors

void pxGaussianBlur(
		matrixImage<float> *buf_in,
		matrixImage<float> *buf_out,
		size_t x,
		size_t y,
		float **ker,
		size_t ker_size,
		const size_t offset)
{
	float px = 0;
	for (size_t k_w = 0; k_w < ker_size; k_w++)
	{
		for (size_t k_h = 0; k_h < ker_size; k_h++)
		{
			if (x + k_w - offset >= buf_in->width ||
				y + k_h - offset >= buf_in->height)
			{
				continue;
			}
			float *px_tmp = buf_in->at(x + k_w - offset, y + k_h - offset);
			float k_elt = ker[k_h][k_w];
			px += *px_tmp * k_elt;
		}
	}
	if (px > 255)
		px = 255;
	buf_out->set(x, y, px);
}

void gaussianBlur(
		matrixImage<float> *buf_in,
		matrixImage<float> *buf_out)
{
	spdlog::info("Gaussian Blurring.");
	size_t width = buf_in->width;
	size_t height = buf_in->height;
	const size_t kernel_size = 7;
	const size_t offset = kernel_size / 2;
	float **kernel = new float *[kernel_size];
	for (size_t i = 0; i < kernel_size; i++)
	{
		kernel[i] = new float[kernel_size];
	}
	kernel = generate_kernel(kernel, kernel_size);
	for (size_t h = 0; h < height; h++)
	{
		for (size_t w = 0; w < width; w++)
		{
			pxGaussianBlur(buf_in, buf_out, w, h, kernel, kernel_size, offset);
		}
	}
}


float pxDilationErosion(matrixImage<float> *matImg,
		const size_t w,
		const size_t h,
		const size_t se_w,
		const size_t se_h,
		const bool d_or_e)
{

	/*
	 * d_or_e == true => dilation, else erosion.
	 */
	float max_value = 0.f;
	float min_value = 256.f;
	size_t off_w = se_w / 2;
	size_t off_h = se_h / 2;
	for (size_t sw = 0; sw + w < matImg->width + off_w && sw < se_w; sw++)
	{
		if (sw + w < off_w)
			continue;
		for (size_t sh = 0; sh + h < matImg->height + off_h && sh < se_h; sh++)
		{
			if (sh + h < off_h)
				continue;
			max_value = std::max(max_value,
					*(matImg->at(sw + w - off_w, sh + h - off_h)));
			min_value = std::min(min_value,
					*(matImg->at(sw + w - off_w, sh + h - off_h)));
		}
	}
	if (d_or_e)
		return max_value;
	return min_value;
}

void dilationErosion(matrixImage<float> *mat_in,
		matrixImage<float> *mat_out,
		const size_t se_w,
		const size_t se_h,
		const bool d_or_e)
{
	/*
	 * d_or_e== true => dilation, else erosion
	 */


	for (size_t w = 0; w < mat_in->width; w++)
	{
		for (size_t h = 0; h < mat_out->height; h++)
		{
			float value = pxDilationErosion(mat_in, w, h, se_w, se_h, d_or_e);
			mat_out->set(w, h, value);
		}
	}
}

void morphOpening(matrixImage<float> *mat_in, matrixImage<float> *mat_out,
		size_t se_w, size_t se_h)
{
	spdlog::info("Morphological Opening");
	dilationErosion(mat_in, mat_out, se_w, se_h, false);
	mat_in->swap(mat_out);
	dilationErosion(mat_in, mat_out, se_w, se_h, true);
}

void morphClosing(matrixImage<float> *mat_in, matrixImage<float> *mat_out,
		size_t se_w, size_t se_h)
{
	spdlog::info("Morphological Closing");
	dilationErosion(mat_in, mat_out, se_w, se_h, true);
	mat_in->swap(mat_out);
	dilationErosion(mat_in, mat_out, se_w, se_h, false);
}

float getAvgIntensity(matrixImage<float> *mat_in)
{
	float avg_intensity = 0.f;
	float nb_iter = 0.f;
	for (size_t w = 0; w < mat_in->width; w++)
	{
		for (size_t h = 0; h < mat_in->height; h++)
		{
			float tmp_px = *(mat_in->at(w, h));
			avg_intensity = avg_intensity * (nb_iter / (nb_iter + 1)) +
							tmp_px / (nb_iter + 1);
			nb_iter++;
		}
	}
	return avg_intensity;
}

void pxBernsenThreshold(matrixImage<float> *mat_in,
						matrixImage<float> *mat_out,
						size_t se_w,
						size_t se_h,
						float contrast_threshold,
						size_t w,
						size_t h,
						size_t off_w,
						size_t off_h,
						float avg_intensity)
{
	float local_contrast;
	float local_midgray = 0.f;
	float value = 0.f; // 0 means it is background, 255 means foreground
	float max_intensity = 0.f;
	float min_intensity = 256.f;
	for (size_t sw = 0; sw < se_w && sw + w < mat_in->width + off_w; sw++)
	{
		if (sw + w < off_w)
			continue;
		for (size_t sh = 0; sh < se_h && sh + h < mat_in->height + off_w; sh++)
		{
			if (sh + h < off_h)
				continue;
			float tmp_px = *(mat_in->at(sw + w - off_w, sh + h - off_h));
			min_intensity = std::min(min_intensity, tmp_px);
			max_intensity = std::max(max_intensity, tmp_px);
		}
	}
	local_contrast = max_intensity - min_intensity;
	local_midgray = (max_intensity + min_intensity) / 2;
	float *cur_px = mat_in->at(w, h);
	if (local_contrast < contrast_threshold)
	{
		if (local_midgray >= avg_intensity)
			value = 255.f;
	}
	else if (*cur_px >= local_midgray)
		value = 255.f;
	mat_out->set(w, h, value);
}

void bernsenThreshold(matrixImage<float> *mat_in,
					  matrixImage<float> *mat_out,
					  size_t se_w,
					  size_t se_h,
					  float contrast_threshold = 15.f)
{
	spdlog::info("Thresholding the image");
	float avg_intensity = getAvgIntensity(mat_in);
	for (size_t w = 0; w < mat_in->width; w++)
	{
		for (size_t h = 0; h < mat_in->height; h++)
		{
			pxBernsenThreshold(mat_in, mat_out, se_w, se_h, contrast_threshold,
							   w, h, se_w / 2, se_h / 2, avg_intensity);
		}
	}
}

void generate_hist(matrixImage<float> *mat_in, int *histo)
{
	for (size_t w = 0; w < mat_in->width; w++)
	{
		for (size_t h = 0; h < mat_in->height; h++)
		{
			float *tmp_px = mat_in->at(w, h);
			int value = (int) (*tmp_px + 0.5);
			histo[value] += 1;
		}
	}
}

int find_mean_intensity(int *histo, int nb_px)
{
	float sigmas[256] = {0.f};
	for (int i = 1; i < 256; i++)
	{
		float wb, wf, mu_b, mu_f, count_b;
		wb = wf = mu_b = mu_f = count_b = 0.f;
		for (int j = 0; j < i; j++)
		{
			count_b += histo[j];
			mu_b += histo[j] * j;
		}
		wb = count_b / (float) nb_px;
		wf = 1.f - wb;
		mu_b /= count_b;
		for (int j = i; j < 256; j++)
		{
			mu_f += histo[j] * j;
		}
		mu_f /= (nb_px - count_b);
		sigmas[i] = wb * wf * std::pow(mu_b - mu_f, 2);
	}
	int max = 0;
	for (int i = 0; i < 256; i++)
	{
		if (sigmas[i] > sigmas[max])
			max = i;
	}
	return max;
}

void otsuThreshold(matrixImage<float> *mat_in, matrixImage<float> *mat_out)
{
	spdlog::info("Otsu thresholding");
	int histo[256] = {0};
	generate_hist(mat_in, histo);
	size_t width = mat_in->width;
	size_t height = mat_in->height;
	int mean_intensity = find_mean_intensity(histo,
											 mat_in->width * mat_in->height);
	spdlog::info("mean_intensity = {}", mean_intensity);
	for (size_t w = 0; w < width; w++)
	{
		for (size_t h = 0; h < height; h++)
		{
			if (*(mat_in->at(w, h)) >= mean_intensity)
				mat_out->set(w, h, 255.f);
			else
				mat_out->set(w, h, 0.f);
		}
	}
}

void update_neighbor(matrixImage<float> *mat_in, matrixImage<float> *mat_out, size_t w, size_t h, std::queue<size_t> &q, float cur_label)
{
	float *value = mat_in->at(w, h);
	float *label = mat_out->at(w, h);
	if (*value != 0.f && *label == 0.f)
	{
		*label = cur_label;
		q.push(h * mat_in->width + w);
	}
}

void update_neighbors(matrixImage<float> *mat_in, matrixImage<float> *mat_out, size_t w, size_t h, std::queue<size_t> &q, float cur_label)
{
	if (w > 0)
		update_neighbor(mat_in, mat_out, w - 1, h, q, cur_label);
	if (w + 1 <  mat_in->width)
		update_neighbor(mat_in, mat_out, w + 1, h, q, cur_label);
	if (h > 0)
		update_neighbor(mat_in, mat_out, w, h - 1, q, cur_label);
	if (h + 1 < mat_in->height)
		update_neighbor(mat_in, mat_out, w, h + 1, q, cur_label);
}

int get_labels(matrixImage<float> *mat_in, matrixImage<float> *mat_out)
{
	spdlog::info("labeling connected components");
	float cur_label = 1.f;
	std::queue<size_t> q;
	for (size_t w = 0; w < mat_in->width; w++)
	{
		for (size_t h = 0; h < mat_in->height; h++)
		{
			if (*(mat_in->at(w, h)) == 0.f || *(mat_out->at(w, h)) != 0.f)
				continue;
			mat_out->set(w, h, cur_label);
			q.push(h * mat_in->width + w);
			while (!q.empty())
			{
				size_t pos = q.front();
				q.pop();
				size_t cur_w = pos % mat_in->width;
				size_t cur_h = pos / mat_in->width;
				update_neighbors(mat_in, mat_out, cur_w, cur_h, q, cur_label);
			}
			cur_label++;
		}
	}
	std::cout << "number of labels is " << cur_label - 1 << std::endl;
	return cur_label - 1;
}

void new_bounding_box(std::vector<std::vector<size_t>> &boundingboxes, size_t w, size_t h)
{
	std::vector<size_t> bbox;
	bbox.push_back(w);
	bbox.push_back(h);
	bbox.push_back(w);
	bbox.push_back(h);
	boundingboxes.push_back(bbox);
}

bool map_contains(std::map<float, int> map, float key)
{
	auto search = map.find(key);
	return search != map.end();
}

void get_bounding_boxes(matrixImage<float> *mat_in, std::vector<std::vector<size_t>> &boundingboxes)
{
	spdlog::info("getting bounding boxes");
	std::map<float, int> met_labels;
	int nb_labels = 0;
	for (size_t w = 0; w < mat_in->width; w++)
	{
		for (size_t h = 0; h < mat_in->height; h++)
		{
			float label = *(mat_in->at(w, h));
			if (label == 0.f)
				continue;
			if (!map_contains(met_labels, label))
			{
				new_bounding_box(boundingboxes, w, h);
				met_labels[label] = nb_labels;
				nb_labels++;
			}
			int bbox_idx = met_labels[label];
			std::vector<size_t> bbox = boundingboxes[bbox_idx];
			if (bbox[0] > w)
				boundingboxes[bbox_idx][0] = w; //upper left x
			if (bbox[2] < w)
				boundingboxes[bbox_idx][2] = w; //lower right x
			if (bbox[1] > h)
				boundingboxes[bbox_idx][1] = h; //upper left y
			if (bbox[3] < h)
				boundingboxes[bbox_idx][3] = h; //lower right y
		}
	}
	for (int i = 0; i < nb_labels; i++)
	{

		boundingboxes[i][2] -= boundingboxes[i][0];
		boundingboxes[i][3] -= boundingboxes[i][1];
	}
}

void useCpu(gil::rgb8_image_t &image1, gil::rgb8_image_t &image2, char *filename, json &bboxes)
{
	matrixImage<uchar3> *matImg1 = toMatrixImage(image1);
	matrixImage<float> *matGray1 = new matrixImage<float>(matImg1->width,
			matImg1->height);
	toGrayscale(matImg1, matGray1);

	matrixImage<uchar3> *matGray1_out = matFloatToMatUchar3(matGray1);
	write_image(matGray1_out, "grayscale_1.png");

	matrixImage<uchar3> *matImg2 = toMatrixImage(image2);
	matrixImage<float> *matGray2 = new matrixImage<float>(matImg2->width,
			matImg2->height);
	toGrayscale(matImg2, matGray2);

	matrixImage<uchar3> *matGray2_out = matFloatToMatUchar3(matGray2);
	write_image(matGray2_out, "grayscale_2.png");

	matrixImage<float> *matGBlur1 = new matrixImage<float>(matImg1->width,
			matImg1->height);
	gaussianBlur(matGray1, matGBlur1);

	matrixImage<uchar3> *matGBlur1_out = matFloatToMatUchar3(matGBlur1);
	write_image(matGBlur1_out, "gaussian_blur_1.png");

	matrixImage<float> *matGBlur2 = new matrixImage<float>(matImg2->width,
			matImg2->height);
	gaussianBlur(matGray2, matGBlur2);

	matrixImage<uchar3> *matGBlur2_out = matFloatToMatUchar3(matGBlur2);
	write_image(matGBlur2_out, "gaussian_blur_2.png");

	matGBlur1->abs_diff(matGBlur2);
	matrixImage<uchar3> *matGBlur1_2_diff_out = matFloatToMatUchar3(matGBlur1);
	write_image(matGBlur1_2_diff_out, "img_abs_diff.png");


	matrixImage<float> *matClosing = new matrixImage<float>(matImg1->width,
															matImg1->height);
	morphClosing(matGBlur1, matClosing, 20, 20);
	matrixImage<uchar3> *matClosing_out = matFloatToMatUchar3(matClosing);
	write_image(matClosing_out, "closing.png");

	matrixImage<float> *matOpening = new matrixImage<float>(matImg1->width,
															matImg1->height);
	morphOpening(matClosing, matOpening, 20, 20);
	matrixImage<uchar3> *matOpening_out = matFloatToMatUchar3(matOpening);
	write_image(matOpening_out, "opening.png");

	matrixImage<float> *matThreshold = new matrixImage<float>(matImg1->width,
															  matImg1->height);
	bernsenThreshold(matOpening, matThreshold, 10, 10);
	matrixImage<uchar3> *matThreshold_out = matFloatToMatUchar3(matThreshold);
	write_image(matThreshold_out, "threshold.png");

	matrixImage<float> *matThreshold2 = new matrixImage<float>(matImg1->width,
															   matImg1->height);
	bernsenThreshold(matThreshold, matThreshold2, 10, 10);
	matrixImage<uchar3> *matThreshold2_out = matFloatToMatUchar3(matThreshold2);
	write_image(matThreshold2_out, "threshold2.png");

	otsuThreshold(matOpening, matThreshold);
	matrixImage<uchar3> *matOtsu_out = matFloatToMatUchar3(matThreshold);
	write_image(matOtsu_out, "otsu.png");

	matrixImage<float> *matLabels = new matrixImage<float>(matImg1->width,matImg1->height);
	matLabels->fill(0.f);
	int nb_labels = get_labels(matThreshold, matLabels);
	matrixImage<uchar3> *matLabels_out = matFloatToMatUchar3(matLabels);
	write_image(matLabels_out, "labels.png");

	std::vector<std::vector<size_t>> boundingboxes;
	get_bounding_boxes(matLabels, boundingboxes);
	for (int i = 0; i < nb_labels; i++)
	{
		std::cout << "[" << boundingboxes[i][0] << ", " << boundingboxes[i][1] << ", " << boundingboxes[i][2] << ", " << boundingboxes[i][3] << "]"<< std::endl;
	}

	std::string f(filename);
	auto base_filename = f.substr(f.find_last_of("/") + 1);
	bboxes[base_filename] = boundingboxes;

	delete matImg1;
	delete matImg2;
	delete matGray1;
	delete matGray2;
	delete matGray1_out;
	delete matGray2_out;
	delete matGBlur1;
	delete matGBlur2;
	delete matGBlur1_out;
	delete matGBlur2_out;
	delete matOpening;
	delete matOpening_out;
	delete matClosing;
	delete matClosing_out;
	delete matThreshold;
	delete matThreshold_out;
}
