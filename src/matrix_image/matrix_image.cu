#include "cudaFunctions/cudaFunctions.cuh"
#include "matrix_image.cuh"
void write_image(matrixImage<uchar3>* matImage, const char* filename)
{
    gil::rgb8c_view_t src = gil::interleaved_view(
        matImage->width, matImage->height,
        (boost::gil::rgb8_pixel_t const*)(matImage->buffer),
        matImage->width * sizeof(uchar3));
    spdlog::info("Writing into {}", filename);
    gil::write_view(filename, src, gil::png_tag());
}

matrixImage<uchar3>* toMatrixImage(gil::rgb8_image_t& image)
{
    gil::rgb8_image_t::const_view_t view = gil::const_view(image);
    assert(view.is_1d_traversable());

    size_t width = view.width();
    size_t height = view.height();
    matrixImage<uchar3>* mat = new matrixImage<uchar3>(width, height);

    for (size_t y = 0; y < height; y += 1)
    {
        auto it = view.row_begin(y);
        for (size_t x = 0; x < width; x++)
        {
            gil::rgb8_pixel_t pixel = it[x];
            uchar3 a = uchar3();
            a.x = gil::at_c<0>(pixel);
            a.y = gil::at_c<1>(pixel);
            a.z = gil::at_c<2>(pixel);
            mat->set(x, y, a);
        }

        // use it[j] to access pixel[i][j]
    }
    return mat;
}

__device__ __host__ uchar3 createUchar3(unsigned char r, unsigned char g,
                                        unsigned char b)
{
    uchar3 ret = uchar3();
    ret.x = r;
    ret.y = g;
    ret.z = b;
    return ret;
}
matrixImage<uchar3>* matFloatToMatUchar3(matrixImage<float>* matIn)
{
    spdlog::info("Converting to uchar3");
    matrixImage<uchar3>* matOut =
        new matrixImage<uchar3>(matIn->width, matIn->height);
    for (size_t w = 0; w < matIn->width; w++)
    {
        for (size_t h = 0; h < matIn->height; h++)
        {
            uchar3 val = uchar3();
            float* valIn = matIn->at(w, h);
            if (*valIn > 255)
            {
                val.x = 255;
                val.y = 255;
                val.z = 255;
            }
            else
            {
                val.x = ceil(*matIn->at(w, h));
                val.y = val.x;
                val.z = val.x;
            }
            matOut->set(w, h, val);
        }
    }
    return matOut;
}
