#include <cstdio>
#include "render_gpu.cuh"
#include "render_cpu.h"
#define USE_GPU true

int main()
{

#if (USE_GPU)
	use_gpu();
#else
	use_cpu();
#endif
	printf("End \n");
	return 0;
}
