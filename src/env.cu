#include <cstdlib>
#include <boost/gil/image.hpp>
#include "env.cuh"

void privateAbortError(const char *msg,const  char *filename,const  char *fname, int line)
{
	spdlog::error("{} ({},file: {}, line: {})", msg, filename, fname, line);
	std::exit(1);
}
