#ifndef _BICUBICINTERPOLATION_CU_
#define _BICUBICINTERPOLATION_CU_

#include"cuda_runtime.h"
#include"device_launch_parameters.h"
#include"curand_kernel.h"
#include<opencv2\opencv.hpp>
#include<opencv2\highgui.hpp>
#include<opencv2\core.hpp>
#include<opencv2\imgproc.hpp>
#include<iostream>
#include<fstream>

using namespace std;
using namespace cv;

inline void checkCudaErrors(cudaError err) //cuda error handle function
{
  if (cudaSuccess != err)
  {
    fprintf(stderr, "CUDA Runtime API error:%s.\n", cudaGetErrorString(err));
    return;
  }
}

__device__ double my_abs(double x)
{
  return x >= 0 ? x : (-x);
}

__device__ double BiCubicFunction(double x)
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

__global__ void bicubicinterpolation(double * cu_src, double * cu_dst, int Width, int Height, int width, int height)
{
  int y = blockDim.x*blockIdx.x + threadIdx.x;
  int x = blockDim.y*blockIdx.y + threadIdx.y;

  if (x <= height&&x >= 0 && y <= width&&y >= 0)
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
          temp += cu_src[distance_x[i]*Width+distance_y[j]]*k_i[i] * k_j[j];
        }
      
      if (temp < 0)
        temp = 0;
      else if (temp > 255)
        temp = 255;

      cu_dst[x*width+y] = temp;
    }
  }
}

__device__ int Round(double x)
{
  int X = (int)x;
  double sub = x - X;
  if (sub >= 0.5)
    return X + 1;
  else return X;
}

__device__ double Sin(double a)
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

__device__ double Cos(double a)
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

__global__ void rotation(double * cu_src, double * cu_dst, int Width, int Height, int width, int height, double angle)
{
  int x = blockDim.y*blockIdx.y + threadIdx.y;
  int y = blockDim.x*blockIdx.x + threadIdx.x;

  if (x >= 0 && x <= height&&y >= 0 && y <= width)
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
            temp += cu_src[distance_x[i] * Width + distance_y[j]] * k_i[i] * k_j[j];
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
      cu_dst[x*width + y] = temp;
    }
  }
}

__global__ void gaussianFilter(double * cu_src, double * cu_dst, int width, int height, double *matrix)
{
  int x = blockDim.y*blockIdx.y + threadIdx.y;
  int y = blockDim.x*blockIdx.x + threadIdx.x;

  if (x < height&&x >= 0 && y < width&&y >= 0)
  {
    if (x - 1 >= 0 && x + 1 <= height - 1 && y - 1 >= 0 && y + 1 <= width - 1)
    {
     cu_src[x*width+y] = cu_src[x*width + y] * matrix[1*3+1] + 
       cu_src[x*width + y-1] * matrix[0*3+1] + 
       cu_src[(x-1)*width + y] * matrix[1*3+0] + 
       cu_src[(x-1)*width + y-1] * matrix[0*3+0] + 
       cu_src[(x-1)*width + y+1] * matrix[2*3+0] + 
       cu_src[x*width + y+1] * matrix[2*3+1] + 
       cu_src[(x+1)*width + y-1] * matrix[0*3+2] + 
       cu_src[(x+1)*width + y] * matrix[1*3+2] + 
       cu_src[(x+1)*width + y+1] * matrix[2*3+2];
    } 
    cu_dst[x*width + y] = cu_src[x*width+y];
  }
  
}

extern "C" void bicubicinterpolation_host(double * src, double * dst, int Width, int Height, int width, int height)
{
  double * inputImage;
  double * outputImage;

  dim3 dimBlock(32, 32);
  dim3 dimGrid((width+dimBlock.x-1) / dimBlock.x, (height+dimBlock.y-1) / dimBlock.y);

  checkCudaErrors(cudaMalloc((void**)&inputImage, sizeof(double)*Width*Height));
  checkCudaErrors(cudaMalloc((void**)&outputImage, sizeof(double)*width*height));

  checkCudaErrors(cudaMemcpy(inputImage, src, sizeof(double)*Width*Height, cudaMemcpyHostToDevice));

  bicubicinterpolation << <dimGrid, dimBlock >> > (inputImage, outputImage, Width, Height, width, height);

  checkCudaErrors(cudaMemcpy(dst, outputImage, sizeof(double)*width*height, cudaMemcpyDeviceToHost));

  cudaFree(inputImage);
  cudaFree(outputImage);
}

extern "C" void rotation_host(double * src, double * dst, int Width, int Height, int width, int height, double angle)
{
  double * inputImage;
  double * outputImage;

  dim3 dimBlock(32, 32);
  dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);

  checkCudaErrors(cudaMalloc((void**)&inputImage, sizeof(double)*Width*Height));
  checkCudaErrors(cudaMalloc((void**)&outputImage, sizeof(double)*width*height));

  checkCudaErrors(cudaMemcpy(inputImage, src, sizeof(double)*Width*Height, cudaMemcpyHostToDevice));

  rotation << < dimGrid, dimBlock >> > (inputImage, outputImage, Width, Height, width, height, angle);

  checkCudaErrors(cudaMemcpy(dst, outputImage, sizeof(double)*width*height, cudaMemcpyDeviceToHost));

  cudaFree(inputImage);
  cudaFree(outputImage);
}

extern "C" void gaussianFilter_host(double * src, double * dst, int width, int height, double *matrix)
{
  double * inputImage;
  double * outputImage;
  double * inmatrix;

  dim3 dimBlock(32, 32);
  dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);

  checkCudaErrors(cudaMalloc((void**)&inputImage, sizeof(double)*width*height));
  checkCudaErrors(cudaMalloc((void**)&outputImage, sizeof(double)*width*height));
  checkCudaErrors(cudaMalloc((void**)&inmatrix, sizeof(double) * 3 * 3));

  checkCudaErrors(cudaMemcpy(inputImage, src, sizeof(double)*width*height, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(inmatrix, matrix, sizeof(double) * 3 * 3, cudaMemcpyHostToDevice));

  gaussianFilter << < dimGrid, dimBlock >> > (inputImage, outputImage, width, height, inmatrix);

  //checkCudaErrors(cudaMemcpy(dst, outputImage, sizeof(double)*width*height, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(dst, outputImage, sizeof(double)*width*height, cudaMemcpyDeviceToHost));

  cudaFree(inputImage);
  cudaFree(outputImage);
}

#endif // !_BICUBICINTERPOLATION_CU_
