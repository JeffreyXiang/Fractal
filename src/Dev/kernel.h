#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void Mandelbrot(uchar4* img, uchar3* palette, int2 size, double4 zone, int max_Iterations);
void Julia(uchar4* img, uchar3* palette, int2 size, double2 con, double4 zone, int max_Iterations);

