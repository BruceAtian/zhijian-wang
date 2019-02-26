#pragma OPENCL EXTENSION cl_nv_printf:enables

__kernel void gaussianFilter_cl(__global double * cl_src, __global double * cl_dst, int width, int height, __global double *matrix)
{
  int y = get_global_id(0);
  int x = get_global_id(1);

  if (x < height&&x >= 0 && y < width&&y >= 0)
  {
    if (x - 1 >= 0 && x + 1 <= height - 1 && y - 1 >= 0 && y + 1 <= width - 1)
    {
      cl_src[x*width + y] = cl_src[x*width + y] * matrix[1 * 3 + 1] +
        cl_src[x*width + y - 1] * matrix[0 * 3 + 1] +
        cl_src[(x - 1)*width + y] * matrix[1 * 3 + 0] +
        cl_src[(x - 1)*width + y - 1] * matrix[0 * 3 + 0] +
        cl_src[(x - 1)*width + y + 1] * matrix[2 * 3 + 0] +
        cl_src[x*width + y + 1] * matrix[2 * 3 + 1] +
        cl_src[(x + 1)*width + y - 1] * matrix[0 * 3 + 2] +
        cl_src[(x + 1)*width + y] * matrix[1 * 3 + 2] +
        cl_src[(x + 1)*width + y + 1] * matrix[2 * 3 + 2];
    }
    cl_dst[x*width + y] = cl_src[x*width + y];
  }
}