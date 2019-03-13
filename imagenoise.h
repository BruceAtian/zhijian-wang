#ifndef IMAGENOISE_H
#define IMAGENOISE_H

#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/opencv.hpp>
#include<vector>
#include<QDebug>
#include<qmessagebox.h>
#include<algorithm>
#include<omp.h>
#include<CL\cl.h>

#define PIXELMAX 255
#define PIXELMIN 0
#define IMAGE_SCALAR 0
#define IMAGE_ROTATE 1
#define PI 3.1415926535

extern "C" void bicubicinterpolation_host(double * src, double * dst, int Width, int Height, int width, int height);
extern "C" void rotation_host(double * src, double * dst, int Width, int Height, int width, int height, double angle);
extern "C" void gaussianFilter_host(double * src, double * dst, int width, int height, double *matrix);

struct Point {
  int x;
  int y;
};

struct Pointf {
  double re;
  double im;
};

class ImageNoise
{
public:
  ImageNoise();
  double generateGaussianNoise(double means, double sigma);//double mu, double sigma);//mu为均值，sigma为方差
  void addGaussianNoise(cv::Mat &img, int k, double means, double sigma, int type);//k为高斯噪声系数
  void addSaultandPepperNoise(cv::Mat &img, double SNR);//SNR为信噪比
  uchar adaptiveMedianFilter_1(cv::Mat &img, int rows, int cols, int kernelsize, int maxsize);//kernelsize为滤波器窗口的尺寸，maxsize为允许滤波器窗口的最大尺寸
  uchar adaptiveMedianFilter_2(cv::Mat &img, int rows, int cols, int kernelsize, int maxsize);
  uchar adaptiveMedianFilter_3(cv::Mat &img, int rows, int cols, int kernelsize, int maxsize);
  void medianFilterResult(cv::Mat &img, int kernelsize, int maxsize);
  void smoothLinearFilter(cv::Mat &img);
  void gaussianFilter(cv::Mat &img, float sigma, int type);
  double calculateNoiseVariance(cv::Mat &src, int kernelSize, double ** means, double ** variance);
  void weinerFilter(cv::Mat &src, int kernelSize);
  void BicubicInterpolation(cv::Mat &src, double fx, double fy, int type, int numofthread);
  void BicubicInterpolation(cv::Mat &src, cv::Mat &dst, double angle, int type, int numofthread);
  double BiCubicFunction(double x);
  void rotate(cv::Mat &img, int theta, int type, int numofthread);
  void BiCubicInterpolation_CL(double * cl_src, double * cl_dst, int Width, int Height, int width, int height);
  void rotationCL(double * cl_src, double * cl_dst, int Width, int Height, int width, int height, double angle);
  void gaussianFilterCL(double * cl_src, double * cl_dst, int width, int height, double *matrix);
  char* LoadProgSource(const char* cFilename, const char* cPreamble, size_t* szFinalLength);
  size_t RoundUp(int groupSize, int globalSize);
};



#endif // IMAGENOISE_H
