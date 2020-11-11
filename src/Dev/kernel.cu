#include "kernel.h"
#include "complex.h"

__global__ void MandelbrotKernel(uchar4* img, uchar3* palette, int2 size, double4 zone, int max_Iterations)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size.x * size.y) return;
    int xIdx = index % size.x;
    int yIdx = index / size.x;
    double x = zone.x * (size.x - xIdx) / size.x + zone.z * xIdx / size.x;
    double y = zone.y * (size.y - yIdx) / size.y + zone.w * yIdx / size.y;
    double2 z = { 0, 0 };
    double2 c = { x, y };
    for (int i = 0; i < max_Iterations; i++)
    {
        z = complexPlus(complexSquare(z), c);
        if (complexLength2(z) > 4.0)
        {
            img[index].x = palette[i % 256].x;
            img[index].y = palette[i % 256].y;
            img[index].z = palette[i % 256].z;
            img[index].w = 255;
            return;
        }
    }
    img[index].x = 0;
    img[index].y = 0;
    img[index].z = 0;
    img[index].w = 255;
}

__global__ void JuliaKernel(uchar4* img, uchar3* palette, int2 size, double2 con, double4 zone, int max_Iterations)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size.x * size.y) return;
    int xIdx = index % size.x;
    int yIdx = index / size.x;
    double x = zone.x * (size.x - xIdx) / size.x + zone.z * xIdx / size.x;
    double y = zone.y * (size.y - yIdx) / size.y + zone.w * yIdx / size.y;
    double2 z = { x, y };
    double2 c = { con.x, con.y };
    for (int i = 0; i < max_Iterations; i++)
    {
        z = complexPlus(complexSquare(z), c);
        if (complexLength2(z) > 4.0)
        {
            img[index].x = palette[i % 256].x;
            img[index].y = palette[i % 256].y;
            img[index].z = palette[i % 256].z;
            img[index].w = 255;
            return;
        }
    }
    img[index].x = 0;
    img[index].y = 0;
    img[index].z = 0;
    img[index].w = 255;
}


void Mandelbrot(uchar4* img, uchar3* palette, int2 size, double4 zone, int max_Iterations)
{
    MandelbrotKernel << <ceil(size.x * size.y / 1024), 1024 >> > (img, palette, size, zone, max_Iterations);
}

void Julia(uchar4* img, uchar3* palette, int2 size, double2 con, double4 zone, int max_Iterations)
{
    JuliaKernel << <ceil(size.x * size.y / 1024), 1024 >> > (img, palette, size, con, zone, max_Iterations);
}
