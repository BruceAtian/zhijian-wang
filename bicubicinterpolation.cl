#pragma OPENCL EXTENSION cl_nv_printf:enables

__OVERLOADABLE__ double my_abs(double x)
{
  return x >= 0 ? x : (-x);
}

__OVERLOADABLE__ double BiCubicFunction(double x)
{
  double a = -0.5;
  double abs_x = my_abs(x);
  if (abs_x <= 1.0)
  {
    return (a + 2)*pow(abs_x, 3) - (a + 3)*pow(abs_x, 2) + 1;
  }
  else if (abs_x < 2.0)
  {
    return a*pow(abs_x, 3) - 5 * a*pow(abs_x, 2) + 8 * a*abs_x - 4 * a;
  }
  else
    return 0.0;
}

__kernel void bicubicinterpolation_cl(__global double * cl_src, __global double * cl_dst, int Width, int Height, int width, int height)
{
  int y = get_global_id(0);
  int x = get_global_id(1);

  if (x < height&&x > 0 && y < width&&y > 0)
  {
    double Xf = (double)x*((double)Height / height);
    double Yf = (double)y*((double)Width / width);
    int X = (int)Xf;//对应源图像的x的整数部分
    int Y = (int)Yf;//对应源图像的y的整数部分

    double k_i[4], k_j[4];
    int distance_x[4], distance_y[4];
    distance_x[0] = X - 1;
    distance_x[1] = X;
    distance_x[2] = X + 1;
    distance_x[3] = X + 2;
    distance_y[0] = Y - 1;
    distance_y[1] = Y;
    distance_y[2] = Y + 1;
    distance_y[3] = Y + 2;
    if ((distance_x[0] >= 0) && (distance_x[3] <= Height) && (distance_y[0] >= 0) && (distance_y[3] <= Width))
    {
      for (int k = 0; k < 4; k++)
      {
        k_i[k] = BiCubicFunction(Xf - distance_x[k]);
        k_j[k] = BiCubicFunction(Yf - distance_y[k]);
      }

      double temp;
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
        {
          temp += cl_src[distance_x[i] * Width + distance_y[j]] * k_i[i] * k_j[j];
        }

      if (temp < 0)
        temp = 0;
      else if (temp > 255)
        temp = 255;

      cl_dst[x*width + y] = temp;
    }
  }
}