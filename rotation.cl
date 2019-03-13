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

__OVERLOADABLE__ int Round(double x)
{
  int X = (int)x;
  double sub = x - X;
  if (sub >= 0.5)
    return X + 1;
  else return X;
}

__OVERLOADABLE__ double Sin(double a)
{
  double pi = 3.1415926535897932;

  double Angle[] = { pi / 4,pi / 8,pi / 16,
    pi / 32,pi / 64,pi / 128,
    pi / 256,pi / 512,pi / 1024,
    pi / 2048,pi / 4096,pi / 8192,pi / 16384 };

  double tang[] = { 1,0.4142135623731,0.19891236737966,
    0.098491403357164,0.049126849769467,
    0.024548622108925,0.012272462379566,
    0.0061360001576234,0.0030679712014227,
    0.0015339819910887,0.00076699054434309,
    0.00038349521577144,0.00019174760083571 };

  if (a <= pi / 16384)
  {
    return a;
  }
  else
  {
    bool ifNegitive = a < 0;
    double x = 10;
    double y = 0;
    double theta = 0;
    for (int i = 0; i < 13; i++)
    {
      double orix, oriy;
      while (theta<a)
      {
        orix = x;
        oriy = y;

        x = orix - tang[i] * oriy;
        y = orix * tang[i] + oriy;
        theta += Angle[i];
      }
      if (theta == a)
      {
        if (ifNegitive)
        {
          return -(y / sqrt(x*x + y*y));
        }
        else
        {
          return (y / sqrt(x*x + y*y));
        }
      }
      else
      {
        theta -= Angle[i];
        x = orix;
        y = oriy;
      }
    }
    if (ifNegitive)
    {
      return -(y / sqrt(x*x + y*y));
    }
    else
    {
      return (y / sqrt(x*x + y*y));
    }
  }
}

__OVERLOADABLE__ double Cos(double a)
{
  double pi = 3.1415926535897932;

  double Angle[] = { pi / 4,pi / 8,pi / 16,
    pi / 32,pi / 64,pi / 128,
    pi / 256,pi / 512,pi / 1024,
    pi / 2048,pi / 4096,pi / 8192,pi / 16384 };

  double tang[] = { 1,0.4142135623731,0.19891236737966,
    0.098491403357164,0.049126849769467,
    0.024548622108925,0.012272462379566,
    0.0061360001576234,0.0030679712014227,
    0.0015339819910887,0.00076699054434309,
    0.00038349521577144,0.00019174760083571 };

  if (a <= pi / 16384)
  {
    return a;
  }
  else
  {
    bool ifNegitive = a < 0;
    double x = 10;
    double y = 0;
    double theta = 0;
    for (int i = 0; i < 13; i++)
    {
      double orix, oriy;
      while (theta<a)
      {
        orix = x;
        oriy = y;

        x = orix - tang[i] * oriy;
        y = orix * tang[i] + oriy;
        theta += Angle[i];
      }
      if (theta == a)
      {
        if (ifNegitive)
        {
          return -(x / sqrt(x*x + y*y));
        }
        else
        {
          return (x / sqrt(x*x + y*y));
        }
      }
      else
      {
        theta -= Angle[i];
        x = orix;
        y = oriy;
      }
    }
    if (ifNegitive)
    {
      return -(x / sqrt(x*x + y*y));
    }
    else
    {
      return (x / sqrt(x*x + y*y));
    }
  }
}

__kernel void rotation_cl(__global double * cl_src, __global double * cl_dst, int Width, int Height, int width, int height, double angle)
{
  int y = get_global_id(0);
  int x = get_global_id(1);

  if (x > 0 && x < height&&y > 0 && y < width)
  {
    double double_X = ((x - height / 2.0f)*Cos(angle)) - ((y - width / 2.0f)*Sin(angle)) + Height / 2.0f;
    double double_Y = ((x - height / 2.0f)*Sin(angle)) + ((y - width / 2.0f)*Cos(angle)) + Width / 2.0f;

    int X = Round((double)((x - height / 2.0f)*Cos(angle)) - ((y - width / 2.0f)*Sin(angle)) + Height / 2.0f);
    int Y = Round((double)((x - height / 2.0f)*Sin(angle)) + ((y - width / 2.0f)*Cos(angle)) + Width / 2.0f);

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
    if ((distance_x[0] >= 0) && (distance_x[3] < Height) && (distance_y[0] >= 0) && (distance_y[3] < Width))
    {
      for (int k = 0; k < 4; k++)
      {
        k_i[k] = BiCubicFunction(double_X - distance_x[k]);
        k_j[k] = BiCubicFunction(double_Y - distance_y[k]);
      }

      double temp = 0;
      if (double_X < 0 || double_X >= Height || double_Y < 0 || double_Y >= Width)
        temp = 0;
      else
      {
        for (int i = 0; i < 3; i++)
          for (int j = 0; j < 3; j++)
          {
            temp += cl_src[distance_x[i] * Width + distance_y[j]] * k_i[i] * k_j[j];
          }
      }
      if (temp >= 256)
      {
        temp = 255;
      }
      else if (temp <= 0)
      {
        temp = 0;
      }
      cl_dst[x*width + y] = temp;
    }
  }
}