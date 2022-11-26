#ifndef GPGPU_ENV_CUH
#define GPGPU_ENV_CUH

#include <spdlog/spdlog.h>
#include <boost/gil/typedefs.hpp>
#include <boost/gil/image.hpp>
#include <boost/gil/extension/io/png/write.hpp>
namespace gil = boost::gil;

void privateAbortError(const char *msg,const char *filename,const char *fname, int line);

#define abortError(msg) privateAbortError(msg,__FILE__, __FUNCTION__, __LINE__)
#endif //GPGPU_ENV_CUH
