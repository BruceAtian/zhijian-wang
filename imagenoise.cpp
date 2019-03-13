#include"imagenoise.h"
#include<iostream>
#include<fstream>
using namespace std;

ImageNoise::ImageNoise()
{

}

double ImageNoise::generateGaussianNoise(double means, double sigma)
{
  static double U1, U2;
  static bool hasSpare = false;

  if (hasSpare)
  {
    hasSpare = false;
    return means + sigma*U1*sin(U2);
  }

  hasSpare = true;

  U1 = (double)rand() / RAND_MAX;
  U2 = (double)rand() / RAND_MAX;

  U1 = sqrt(-2 * log(U1));
  U2 = 2 * PI*U2;

  return means + sigma*U1*cos(U2);
}

void ImageNoise::addGaussianNoise(cv::Mat &img, int k, double means, double sigma, int type)
{
  for (int x = 0; x < img.rows; x++)
  {
    for (int y = 0; y < (img.cols*img.channels()); y++)
    {
      double temp = img.at<uchar>(x, y) + k*generateGaussianNoise(means, sigma);
      if (temp > PIXELMAX)
        temp = PIXELMAX;
      else if (temp < PIXELMIN)
        temp = PIXELMIN;
      img.at<uchar>(x, y) = temp;
    }
  }
}

void ImageNoise::addSaultandPepperNoise(cv::Mat &img, double SNR)
{
  int SP = img.rows*img.cols;
  int NP = SP*(1 - SNR);

  for (int i = 0; i < NP; i++)
  {
    int x = (int)(rand()*1.0 / RAND_MAX* (double)img.rows);
    int y = (int)(rand()*1.0 / RAND_MAX* (double)img.cols);
    int r = rand() % 2;
    if (x >= 0 && x < img.rows && y >= 0 && y < img.cols)
    {
      if (r)
      {
        img.at<cv::Vec3b>(x, y) = cv::Vec3b(0, 0, 0);
      }
      else
      {

        img.at<cv::Vec3b>(x, y) = cv::Vec3b(255, 255, 255);
      }
    }
  }
}

uchar ImageNoise::adaptiveMedianFilter_1(cv::Mat &img, int rows, int cols, int kernelsize, int maxsize)
{
  uchar Zmin;
  uchar Zmed;
  uchar Zmax;
  uchar Zxy;
  uchar outPixels;

  std::vector<uchar>pixels;
  do {
    for (int x = -kernelsize / 2; x <= kernelsize / 2; x++)
    {
      for (int y = -kernelsize / 2; y <= kernelsize / 2; y++)
      {
        pixels.push_back(img.at<cv::Vec3b>(rows + x, cols + y)[0]);
      }
    }

    sort(pixels.begin(), pixels.end());
    Zmin = pixels[0];
    Zmed = pixels[kernelsize*kernelsize / 2];
    Zmax = pixels[kernelsize*kernelsize - 1];
    Zxy = img.at<cv::Vec3b>(rows, cols)[0];

    if (Zmed<Zmax && Zmed>Zmin)
    {
      if (Zxy<Zmax && Zxy>Zmin)
      {
        outPixels = Zxy;
        break;
      }
      else
      {
        outPixels = Zmed;
        break;
      }
    }
    else
    {
      kernelsize += 2;
    }
  } while (kernelsize <= maxsize);

  if (kernelsize > maxsize)
    outPixels = Zmed;

  return outPixels;
}


uchar ImageNoise::adaptiveMedianFilter_2(cv::Mat &img, int rows, int cols, int kernelsize, int maxsize)
{
  uchar Zmin;
  uchar Zmed;
  uchar Zmax;
  uchar Zxy;
  uchar outPixels;

  std::vector<uchar>pixels;
  do {
    for (int x = -kernelsize / 2; x <= kernelsize / 2; x++)
    {
      for (int y = -kernelsize / 2; y <= kernelsize / 2; y++)
      {
        pixels.push_back(img.at<cv::Vec3b>(rows + x, cols + y)[1]);
      }
    }

    sort(pixels.begin(), pixels.end());
    Zmin = pixels[0];
    Zmed = pixels[kernelsize*kernelsize / 2];
    Zmax = pixels[kernelsize*kernelsize - 1];
    Zxy = img.at<cv::Vec3b>(rows, cols)[1];

    if (Zmed<Zmax && Zmed>Zmin)
    {
      if (Zxy<Zmax && Zxy>Zmin)
      {
        outPixels = Zxy;
        break;
      }
      else
      {
        outPixels = Zmed;
        break;
      }
    }
    else
    {
      kernelsize += 2;
    }
  } while (kernelsize <= maxsize);

  if (kernelsize > maxsize)
    outPixels = Zmed;

  return outPixels;
}


uchar ImageNoise::adaptiveMedianFilter_3(cv::Mat &img, int rows, int cols, int kernelsize, int maxsize)
{
  uchar Zmin;
  uchar Zmed;
  uchar Zmax;
  uchar Zxy;
  uchar outPixels;

  std::vector<uchar>pixels;
  do {
    for (int x = -kernelsize / 2; x <= kernelsize / 2; x++)
    {
      for (int y = -kernelsize / 2; y <= kernelsize / 2; y++)
      {
        pixels.push_back(img.at<cv::Vec3b>(rows + x, cols + y)[2]);
      }
    }

    sort(pixels.begin(), pixels.end());
    Zmin = pixels[0];
    Zmed = pixels[kernelsize*kernelsize / 2];
    Zmax = pixels[kernelsize*kernelsize - 1];
    Zxy = img.at<cv::Vec3b>(rows, cols)[2];

    if (Zmed<Zmax && Zmed>Zmin)
    {
      if (Zxy<Zmax && Zxy>Zmin)
      {
        outPixels = Zxy;
        break;
      }
      else
      {
        outPixels = Zmed;
        break;
      }
    }
    else
    {
      kernelsize += 2;
    }
  } while (kernelsize <= maxsize);

  if (kernelsize > maxsize)
    outPixels = Zmed;

  return outPixels;
}


void ImageNoise::medianFilterResult(cv::Mat &img, int kernelsize, int maxsize)
{
  cv::copyMakeBorder(img, img, maxsize / 2, maxsize / 2, maxsize / 2, maxsize / 2, cv::BorderTypes::BORDER_REPLICATE);

  for (int x = maxsize / 2; x < img.rows - maxsize / 2; x++)
  {
    for (int y = maxsize / 2; y < img.cols - maxsize / 2; y++)
    {
      img.at<cv::Vec3b>(x, y)[0] = adaptiveMedianFilter_1(img, x, y, kernelsize, maxsize);
      img.at<cv::Vec3b>(x, y)[1] = adaptiveMedianFilter_2(img, x, y, kernelsize, maxsize);
      img.at<cv::Vec3b>(x, y)[2] = adaptiveMedianFilter_3(img, x, y, kernelsize, maxsize);
    }
  }

  cv::Mat mask(img, cv::Rect(maxsize / 2, maxsize / 2, img.cols - maxsize*0.8, img.rows - maxsize*0.8));
  img = mask.clone();
}


void ImageNoise::smoothLinearFilter(cv::Mat &img)
{
  int filter[3][3] = {
      {1, 2, 1},
      {2, 4, 2},
      {1, 2, 1}
  };
  int weightSum = 16;
  int filterWidth = 3;
  cv::copyMakeBorder(img, img, filterWidth - 1, filterWidth - 1, filterWidth - 1, filterWidth - 1, cv::BorderTypes::BORDER_REPLICATE);

  for (int x = filterWidth - 1; x < img.rows - (filterWidth - 1); x++)
  {
    for (int y = 2; y < img.cols - (filterWidth - 1); y++)
    {
      for (int i = 0; i < 3; i++)
      {
        img.at<cv::Vec3b>(x, y)[i] = (img.at<cv::Vec3b>(x, y)[i] * filter[filterWidth / 2][filterWidth / 2] +
          img.at<cv::Vec3b>(x, y - 1)[i] * filter[filterWidth / 2][filterWidth / 2 - 1] +
          img.at<cv::Vec3b>(x - 1, y)[i] * filter[filterWidth / 2 - 1][filterWidth / 2] +
          img.at<cv::Vec3b>(x - 1, y - 1)[i] * filter[filterWidth / 2 - 1][filterWidth / 2 - 1] +
          img.at<cv::Vec3b>(x - 1, y + 1)[i] * filter[filterWidth / 2 - 1][filterWidth / 2 + 1] +
          img.at<cv::Vec3b>(x, y + 1)[i] * filter[filterWidth / 2][filterWidth / 2 + 1] +
          img.at<cv::Vec3b>(x + 1, y - 1)[i] * filter[filterWidth / 2 + 1][filterWidth / 2 - 1] +
          img.at<cv::Vec3b>(x + 1, y)[i] * filter[filterWidth / 2 + 1][filterWidth / 2] +
          img.at<cv::Vec3b>(x + 1, y + 1)[i] * filter[filterWidth / 2 + 1][filterWidth / 2 + 1]) / weightSum;
      }
    }
  }

  cv::Mat mask(img, cv::Rect(filterWidth - 1, filterWidth - 1, img.cols - filterWidth*1.1, img.rows - filterWidth*1.1));
  img = mask.clone();
}


void ImageNoise::gaussianFilter(cv::Mat &img, float sigma, int type)
{
  Point coordinateMatrix[3][3] = {
      {(-1, -1), (0, -1), (1, -1)},
      {(-1, 0), (0, 0), (1, 0)},
      {(-1, 1), (0, 1), (1, 1)}
  };
  float weightMatrix[3][3];
  float sum = 0;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
    {
      weightMatrix[i][j] = exp(-(pow(coordinateMatrix[i][j].x, 2) + pow(coordinateMatrix[i][j].y, 2))
        / (2 * sigma*sigma)) / (2 * PI*sigma*sigma);//根据公式计算出当前的加权值
      sum += weightMatrix[i][j];//计算初步求得的加权值的和
    }

  for (int m = 0; m < 3; m++)
    for (int n = 0; n < 3; n++)
    {
      weightMatrix[m][n] *= 1 / sum;//计算真正的加权值
    }

  if (type == 1)
  {
    cv::copyMakeBorder(img, img, 2, 2, 2, 2, cv::BorderTypes::BORDER_REPLICATE);

    for (int x = 2; x < img.rows - 2; x++)
      for (int y = 2; y < img.cols - 2; y++)
      {
        for (int i = 0; i < 3; i++)
          img.at<cv::Vec3b>(x, y)[i] = img.at<cv::Vec3b>(x, y)[i] * weightMatrix[1 + 0][1 + 0] +
          img.at<cv::Vec3b>(x, y - 1)[i] * weightMatrix[1 - 1][1 + 0] +
          img.at<cv::Vec3b>(x - 1, y)[i] * weightMatrix[1 + 0][1 - 1] +
          img.at<cv::Vec3b>(x - 1, y - 1)[i] * weightMatrix[1 - 1][1 - 1] +
          img.at<cv::Vec3b>(x - 1, y + 1)[i] * weightMatrix[1 + 1][1 - 1] +
          img.at<cv::Vec3b>(x, y + 1)[i] * weightMatrix[1 + 1][1 + 0] +
          img.at<cv::Vec3b>(x + 1, y - 1)[i] * weightMatrix[1 - 1][1 + 1] +
          img.at<cv::Vec3b>(x + 1, y)[i] * weightMatrix[1 + 0][1 + 1] +
          img.at<cv::Vec3b>(x + 1, y + 1)[i] * weightMatrix[1 + 1][1 + 1];
      }

    cv::Mat mask(img, cv::Rect(2, 2, img.cols - 2 * 2.0, img.rows - 2 * 2.0));
    img = mask.clone();
  }
  else if (type == 4)
  {
    int width = img.cols;
    int height = img.rows;
    double * src = (double*)malloc(sizeof(double)*width*height);
    double * dst = (double*)malloc(sizeof(double)*width*height);
    double * srcR = (double*)malloc(sizeof(double)*width*height);
    double * srcG = (double*)malloc(sizeof(double)*width*height);
    double * srcB = (double*)malloc(sizeof(double)*width*height);
    double * dstR = (double*)malloc(sizeof(double)*width*height);
    double * dstB = (double*)malloc(sizeof(double)*width*height);
    double * dstG = (double*)malloc(sizeof(double)*width*height);
    double * matrix = (double*)malloc(sizeof(double) * 3 * 3);

    for (int i = 0; i < 3; i++)
    {
      for (int j = 0; j < 3; j++)
      {
        matrix[i * 3 + j] = (double)weightMatrix[i][j];
      }
    }
    if (img.channels() == 1)
    {
      for (int i = 0; i < height; i++)
      {
        for (int j = 0; j < width; j++)
        {
          src[i*width + j] = (double)img.at<uchar>(i, j);
        }
      }

      gaussianFilter_host(src, dst, width, height, matrix);

      for (int i = 0; i < height; i++)
      {
        for (int j = 0; j < width; j++)
        {
          img.at<uchar>(i, j) = dst[i*width + j];
        }
      }
    }
    else if (img.channels() == 3)
    {
      for (int i = 0; i < height; i++)
      {
        for (int j = 0; j < width; j++)
        {
          srcR[i*width + j] = (double)img.at<cv::Vec3b>(i, j)[0];
          srcG[i*width + j] = (double)img.at<cv::Vec3b>(i, j)[1];
          srcB[i*width + j] = (double)img.at<cv::Vec3b>(i, j)[2];
        }
      }

      gaussianFilter_host(srcR, dstR, width, height, matrix);
      gaussianFilter_host(srcG, dstG, width, height, matrix);
      gaussianFilter_host(srcB, dstB, width, height, matrix);

      for (int i = 0; i < height; i++)
      {
        for (int j = 0; j < width; j++)
        {
          img.at<cv::Vec3b>(i, j)[0] = dstR[i*width + j];
          img.at<cv::Vec3b>(i, j)[1] = dstG[i*width + j];
          img.at<cv::Vec3b>(i, j)[2] = dstB[i*width + j];
        }
      }
    }
  }
  else if (type == 5)
  {
    int width = img.cols;
    int height = img.rows;
    double * src = (double*)malloc(sizeof(double)*width*height);
    double * dst = (double*)malloc(sizeof(double)*width*height);
    double * srcR = (double*)malloc(sizeof(double)*width*height);
    double * srcG = (double*)malloc(sizeof(double)*width*height);
    double * srcB = (double*)malloc(sizeof(double)*width*height);
    double * dstR = (double*)malloc(sizeof(double)*width*height);
    double * dstB = (double*)malloc(sizeof(double)*width*height);
    double * dstG = (double*)malloc(sizeof(double)*width*height);
    double * matrix = (double*)malloc(sizeof(double) * 3 * 3);

    for (int i = 0; i < 3; i++)
    {
      for (int j = 0; j < 3; j++)
      {
        matrix[i * 3 + j] = (double)weightMatrix[i][j];
      }
    }
    if (img.channels() == 1)
    {
      for (int i = 0; i < height; i++)
      {
        for (int j = 0; j < width; j++)
        {
          src[i*width + j] = (double)img.at<uchar>(i, j);
        }
      }

      gaussianFilterCL(src, dst, width, height, matrix);

      for (int i = 0; i < height; i++)
      {
        for (int j = 0; j < width; j++)
        {
          img.at<uchar>(i, j) = dst[i*width + j];
        }
      }
    }
    else if (img.channels() == 3)
    {
      for (int i = 0; i < height; i++)
      {
        for (int j = 0; j < width; j++)
        {
          srcR[i*width + j] = (double)img.at<cv::Vec3b>(i, j)[0];
          srcG[i*width + j] = (double)img.at<cv::Vec3b>(i, j)[1];
          srcB[i*width + j] = (double)img.at<cv::Vec3b>(i, j)[2];
        }
      }

      gaussianFilterCL(srcR, dstR, width, height, matrix);
      gaussianFilterCL(srcG, dstG, width, height, matrix);
      gaussianFilterCL(srcB, dstB, width, height, matrix);

      for (int i = 0; i < height; i++)
      {
        for (int j = 0; j < width; j++)
        {
          img.at<cv::Vec3b>(i, j)[0] = dstR[i*width + j];
          img.at<cv::Vec3b>(i, j)[1] = dstG[i*width + j];
          img.at<cv::Vec3b>(i, j)[2] = dstB[i*width + j];
        }
      }
    }
  }
}

double ImageNoise::calculateNoiseVariance(cv::Mat &src, int kernelSize, double **means, double **variance)
{
  int width = src.cols;
  int height = src.rows;
  int center_pos = kernelSize / 2;
  double sumvariance = 0.0;
  double ** dst = new double*[width*height];

  for (int i = 0; i < width*height; i++)
  {
    dst[i] = new double[3];
  }

  for (int i = 0; i < height; i++)
  {
    for (int j = 0; j < width; j++)
    {
      for (int k = 0; k < src.channels(); k++)
      {
        dst[i*width + j][k] = (double)(src.at<cv::Vec3b>(i, j)[k]);
      }
    }
  }

  for (int k = 0; k < src.channels(); k++)
  {
    for (int i = 0; i < height; i++)
    {
      for (int j = 0; j < width; j++)
      {
        for (int m = 0; m < kernelSize; m++)
        {
          for (int n = 0; n < kernelSize; n++)
          {
            int indexI = i + (m - center_pos);
            int indexJ = j + (n - center_pos);
            if (indexI >= 0 && indexI < height && indexJ >= 0 && indexJ < width)
            {
              means[i*width + j][k] += dst[indexI*width + indexJ][k];
              variance[i*width + j][k] += dst[indexI*width + indexJ][k] * dst[indexI*width + indexJ][k];
            }
          }
        }
        means[i*width + j][k] = means[i*width + j][k] / (kernelSize*kernelSize);
        variance[i*width + j][k] = variance[i*width + j][k] / (kernelSize*kernelSize);
        variance[i*width + j][k] -= means[i*width + j][k] * means[i*width + j][k];
        sumvariance += variance[i*width + j][k];
      }
    }
  }
  delete dst;
  return sumvariance / (width*height*src.channels());
}

void ImageNoise::weinerFilter(cv::Mat &src, int kernelSize)
{
  cv::copyMakeBorder(src, src, kernelSize - 1, kernelSize - 1, kernelSize - 1, kernelSize - 1, cv::BorderTypes::BORDER_REPLICATE);
  int width = src.cols;
  int height = src.rows;
  double ** localMeans = new double*[width*height];
  double ** localVariance = new double*[width*height];
  double ** temp = new double*[width*height];
  for (int i = 0; i < width*height; i++)
  {
    temp[i] = new double[3];
  }

  for (int i = 0; i < height; i++)
  {
    for (int j = 0; j < width; j++)
    {
      for (int k = 0; k < src.channels(); k++)
      {
        temp[i*width + j][k] = (double)(src.at<cv::Vec3b>(i, j)[k]);
      }
    }
  }
  for (int i = 0; i < width*height; i++)
  {
    localMeans[i] = new double[3];
    localVariance[i] = new double[3];
  }
  double * dst = new double[width*height];

  //初始化localMeans和localVariance
  for (int i = 0; i < height*width; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      localMeans[i][j] = 0.0;
      localVariance[i][j] = 0.0;
    }
  }

  double noiseVariance = calculateNoiseVariance(src, kernelSize, localMeans, localVariance);
  for (int k = 0; k < src.channels(); k++)
  {
    for (int i = 0; i < height; i++)
    {
      for (int j = 0; j < width; j++)
      {
        dst[i*width + j] = temp[i*width + j][k] - localMeans[i*width + j][k];//f = g-localmeans
        temp[i*width + j][k] = localVariance[i*width + j][k] - noiseVariance;//g = localVariance - noise
        if (temp[i*width + j][k] < 0.0)
        {
          temp[i*width + j][k] = 0.0;
        }
        localVariance[i*width + j][k] = cv::max(localVariance[i*width + j][k], noiseVariance);
        dst[i*width + j] = dst[i*width + j] * temp[i*width + j][k] / localVariance[i*width + j][k] + localMeans[i*width + j][k];
        temp[i*width + j][k] = dst[i*width + j];
      }
    }
  }

  delete localMeans;
  delete localVariance;
  delete dst;

  for (int i = 0; i < height; i++)
  {
    for (int j = 0; j < width; j++)
    {
      for (int k = 0; k < src.channels(); k++)
      {
        src.at<cv::Vec3b>(i, j)[k] = temp[i*width + j][k];
      }
    }
  }

  cv::Mat mask(src, cv::Rect(kernelSize - 1, kernelSize - 1, src.cols - kernelSize*1.5, src.rows - kernelSize*1.5));
  src = mask.clone();

  delete temp;
}

double ImageNoise::BiCubicFunction(double x)
{
  double a = -0.5;
  double abs_x = abs(x);
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

void ImageNoise::BicubicInterpolation(cv::Mat &src, double fx, double fy, int type, int numofthread)
{
  cv::Mat dst;
  dst.create(src.rows*fx, src.cols*fy, src.type());
  if (type == 1)
  {
    for (int x = 0; x < dst.rows; x++)
      for (int y = 0; y < dst.cols; y++)
      {
        int X = (int)(x / fx);//对应源图像的x的整数部分
        int Y = (int)(y / fy);//对应源图像的y的整数部分

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
        if ((distance_x[0] >= 0) && (distance_x[3] < src.rows) && (distance_y[0] >= 0) && (distance_y[3] < src.cols))
        {
          for (int k = 0; k < 4; k++)
          {
            k_i[k] = BiCubicFunction(x / fx - distance_x[k]);
            k_j[k] = BiCubicFunction(y / fy - distance_y[k]);
          }
          if (src.channels() == 1)
          {
            double temp = 0;
            for (int i = 0; i < 3; i++)
              for (int j = 0; j < 3; j++)
              {
                temp = temp + (double)src.at<uchar>(distance_x[i], distance_y[j])*k_i[i] * k_j[j];
              }

            if (temp < 0)
              temp = 0;
            else if (temp > 255)
              temp = 255;

            dst.at<uchar>(x, y) = (uchar)temp;
          }

          if (src.channels() == 3)
          {
            cv::Vec3f temp = { 0,0,0 };
            for (int i = 0; i < 3; i++)
              for (int j = 0; j < 3; j++)
              {
                temp = temp + (cv::Vec3f)src.at<cv::Vec3b>(distance_x[i], distance_y[j])*k_i[i] * k_j[j];
              }
            for (int m = 0; m < 3; m++)
            {
              if (temp[m] < 0)
                temp[m] = 0;
              else if (temp[m] > 255)
                temp[m] = 255;
            }
            dst.at<cv::Vec3b>(x, y) = (cv::Vec3b)temp;
          }
        }

      }
  }
  else if (type == 3)
  {
    omp_set_num_threads(numofthread);
#pragma omp parallel for
    for (int x = 0; x < dst.rows; x++)
      for (int y = 0; y < dst.cols; y++)
      {
        int X = (int)(x / fx);//对应源图像的x的整数部分
        int Y = (int)(y / fy);//对应源图像的y的整数部分

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
        if ((distance_x[0] >= 0) && (distance_x[3] < src.rows) && (distance_y[0] >= 0) && (distance_y[3] < src.cols))
        {
          for (int k = 0; k < 4; k++)
          {
            k_i[k] = BiCubicFunction(x / fx - distance_x[k]);
            k_j[k] = BiCubicFunction(y / fy - distance_y[k]);

          }
          if (src.channels() == 1)
          {
            double temp = 0;
            for (int i = 0; i < 3; i++)
              for (int j = 0; j < 3; j++)
              {
                temp = temp + (double)src.at<uchar>(distance_x[i], distance_y[j])*k_i[i] * k_j[j];
              }

            if (temp < 0)
              temp = 0;
            else if (temp > 255)
              temp = 255;

            dst.at<uchar>(x, y) = (uchar)temp;
          }

          if (src.channels() == 3)
          {
            cv::Vec3f temp = { 0,0,0 };
            for (int i = 0; i < 3; i++)
              for (int j = 0; j < 3; j++)
              {
                temp = temp + (cv::Vec3f)src.at<cv::Vec3b>(distance_x[i], distance_y[j])*k_i[i] * k_j[j];
              }
            for (int m = 0; m < 3; m++)
            {
              if (temp[m] < 0)
                temp[m] = 0;
              else if (temp[m] > 255)
                temp[m] = 255;
            }
            dst.at<cv::Vec3b>(x, y) = (cv::Vec3b)temp;
          }
        }

      }
  }
  else if (type == 4)
  {
    int width = dst.cols;
    int height = dst.rows;
    int Width = src.cols;
    int Height = src.rows;
    double * pixel = (double*)malloc(sizeof(double)*src.rows*src.cols);
    double * dst_pixel = (double*)malloc(sizeof(double)*dst.rows*dst.cols);
    double * pixelR = (double*)malloc(sizeof(double)*src.rows*src.cols);
    double * pixelG = (double*)malloc(sizeof(double)*src.rows*src.cols);
    double * pixelB = (double*)malloc(sizeof(double)*src.rows*src.cols);
    double * dst_pixelR = (double*)malloc(sizeof(double)*dst.rows*dst.cols);
    double * dst_pixelG = (double*)malloc(sizeof(double)*dst.rows*dst.cols);
    double * dst_pixelB = (double*)malloc(sizeof(double)*dst.rows*dst.cols);

    if (src.channels() == 1)
    {
      for (int i = 0; i < src.rows; i++)
      {
        for (int j = 0; j < src.cols; j++)
        {
          pixel[i*src.cols + j] = (double)src.at<uchar>(i, j);
        }
      }
      for (int i = 0; i < dst.rows; i++)
      {
        for (int j = 0; j < dst.cols; j++)
        {
          dst_pixel[i*dst.cols + j] = 0;
        }
      }

      bicubicinterpolation_host(pixel, dst_pixel, Width, Height, width, height);
      
      for (int i = 0; i < dst.rows; i++)
      {
        for (int j = 0; j < dst.cols; j++)
        {
          dst.at<uchar>(i, j) = (uchar)dst_pixel[i*dst.cols + j];
        }
      }
    }
    else if (src.channels() == 3)
    {
      for (int i = 0; i < src.rows; i++)
      {
        for (int j = 0; j < src.cols; j++)
        {
          pixelR[i*src.cols + j] = (double)src.at<cv::Vec3b>(i, j)[0];
          pixelG[i*src.cols + j] = (double)src.at<cv::Vec3b>(i, j)[1];
          pixelB[i*src.cols + j] = (double)src.at<cv::Vec3b>(i, j)[2];
        }
      }

      for (int i = 0; i < dst.rows; i++)
      {
        for (int j = 0; j < dst.cols; j++)
        {
          dst_pixelR[i*dst.cols + j] = 0;
          dst_pixelG[i*dst.cols + j] = 0;
          dst_pixelB[i*dst.cols + j] = 0;
        }
      }

      bicubicinterpolation_host(pixelR, dst_pixelR, Width, Height, width, height);
      bicubicinterpolation_host(pixelG, dst_pixelG, Width, Height, width, height);
      bicubicinterpolation_host(pixelB, dst_pixelB, Width, Height, width, height);

      for (int i = 0; i < dst.rows; i++)
      {
        for (int j = 0; j < dst.cols; j++)
        {
          dst.at<cv::Vec3b>(i, j)[0] = dst_pixelR[i*dst.cols + j];
          dst.at<cv::Vec3b>(i, j)[1] = dst_pixelG[i*dst.cols + j];
          dst.at<cv::Vec3b>(i, j)[2] = dst_pixelB[i*dst.cols + j];
        }
      }
      
    }
  }
  else if (type == 5)
  {
    int width = dst.cols;
    int height = dst.rows;
    int Width = src.cols;
    int Height = src.rows;
    double * pixel = (double*)malloc(sizeof(double)*src.rows*src.cols);
    double * dst_pixel = (double*)malloc(sizeof(double)*dst.rows*dst.cols);
    double * pixelR = (double*)malloc(sizeof(double)*src.rows*src.cols);
    double * pixelG = (double*)malloc(sizeof(double)*src.rows*src.cols);
    double * pixelB = (double*)malloc(sizeof(double)*src.rows*src.cols);
    double * dst_pixelR = (double*)malloc(sizeof(double)*dst.rows*dst.cols);
    double * dst_pixelG = (double*)malloc(sizeof(double)*dst.rows*dst.cols);
    double * dst_pixelB = (double*)malloc(sizeof(double)*dst.rows*dst.cols);

    if (src.channels() == 1)
    {
      for (int i = 0; i < src.rows; i++)
      {
        for (int j = 0; j < src.cols; j++)
        {
          pixel[i*src.cols + j] = (double)src.at<uchar>(i, j);
        }
      }
      for (int i = 0; i < dst.rows; i++)
      {
        for (int j = 0; j < dst.cols; j++)
        {
          dst_pixel[i*dst.cols + j] = 0;
        }
      }

      BiCubicInterpolation_CL(pixel, dst_pixel, Width, Height, width, height);

      for (int i = 0; i < dst.rows; i++)
      {
        for (int j = 0; j < dst.cols; j++)
        {
          dst.at<uchar>(i, j) = (uchar)dst_pixel[i*dst.cols + j];
        }
      }
    }
    else if (src.channels() == 3)
    {
      for (int i = 0; i < src.rows; i++)
      {
        for (int j = 0; j < src.cols; j++)
        {
          pixelR[i*src.cols + j] = (double)src.at<cv::Vec3b>(i, j)[0];
          pixelG[i*src.cols + j] = (double)src.at<cv::Vec3b>(i, j)[1];
          pixelB[i*src.cols + j] = (double)src.at<cv::Vec3b>(i, j)[2];
        }
      }

      for (int i = 0; i < dst.rows; i++)
      {
        for (int j = 0; j < dst.cols; j++)
        {
          dst_pixelR[i*dst.cols + j] = 0;
          dst_pixelG[i*dst.cols + j] = 0;
          dst_pixelB[i*dst.cols + j] = 0;
        }
      }

      BiCubicInterpolation_CL(pixelR, dst_pixelR, Width, Height, width, height);
      BiCubicInterpolation_CL(pixelG, dst_pixelG, Width, Height, width, height);
      BiCubicInterpolation_CL(pixelB, dst_pixelB, Width, Height, width, height);

      for (int i = 0; i < dst.rows; i++)
      {
        for (int j = 0; j < dst.cols; j++)
        {
          dst.at<cv::Vec3b>(i, j)[0] = dst_pixelR[i*dst.cols + j];
          dst.at<cv::Vec3b>(i, j)[1] = dst_pixelG[i*dst.cols + j];
          dst.at<cv::Vec3b>(i, j)[2] = dst_pixelB[i*dst.cols + j];
        }
      }

    }
  }
  cv::imshow("output", dst);
}


void ImageNoise::BicubicInterpolation(cv::Mat &src, cv::Mat &dst, double angle, int type, int numofthread)
{
  if (type == 1)
  {
    for (int x = 0; x < dst.rows; x++)
      for (int y = 0; y < dst.cols; y++)
      {
        double double_X = ((x - dst.rows / 2.0f)*cos(angle)) - ((y - dst.cols / 2.0f)*sin(angle)) + src.rows / 2.0f;
        double double_Y = ((x - dst.rows / 2.0f)*sin(angle)) + ((y - dst.cols / 2.0f)*cos(angle)) + src.cols / 2.0f;

        int X = round((double)((x - dst.rows / 2.0f)*cos(angle)) - ((y - dst.cols / 2.0f)*sin(angle)) + src.rows / 2.0f);
        int Y = round((double)((x - dst.rows / 2.0f)*sin(angle)) + ((y - dst.cols / 2.0f)*cos(angle)) + src.cols / 2.0f);

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
        if ((distance_x[0] >= 0) && (distance_x[3] < src.rows) && (distance_y[0] >= 0) && (distance_y[3] < src.cols))
        {
          for (int k = 0; k < 4; k++)
          {
            k_i[k] = BiCubicFunction(double_X - distance_x[k]);
            k_j[k] = BiCubicFunction(double_Y - distance_y[k]);
          }
          if (src.channels() == 1)
          {
            double temp = 0;
            if (double_X < 0 || double_X >= src.rows || double_Y < 0 || double_Y >= src.cols)
              temp = 0;
            else
            {
              for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                {
                  temp = temp + (double)src.at<uchar>(distance_x[i], distance_y[j])*k_i[i] * k_j[j];
                }
            }

            dst.at<uchar>(x, y) = (uchar)temp;
          }

          if (src.channels() == 3)
          {
            cv::Vec3f temp = { 0,0,0 };
            if (double_X < 0 || double_X >= src.rows || double_Y < 0 || double_Y >= src.cols)
              temp = { 0,0,0 };
            else
            {
              for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                {
                  temp = temp + (cv::Vec3f)src.at<cv::Vec3b>(distance_x[i], distance_y[j])*k_i[i] * k_j[j];
                }
            }
            dst.at<cv::Vec3b>(x, y) = (cv::Vec3b)temp;
          }
        }
      }
  }
  else if (type == 3)
  {
    omp_set_num_threads(numofthread);
#pragma omp parallel for
    for (int x = 0; x < dst.rows; x++)
      for (int y = 0; y < dst.cols; y++)
      {
        double double_X = ((x - dst.rows / 2.0f)*cos(angle)) - ((y - dst.cols / 2.0f)*sin(angle)) + src.rows / 2.0f;
        double double_Y = ((x - dst.rows / 2.0f)*sin(angle)) + ((y - dst.cols / 2.0f)*cos(angle)) + src.cols / 2.0f;

        int X = round((double)((x - dst.rows / 2.0f)*cos(angle)) - ((y - dst.cols / 2.0f)*sin(angle)) + src.rows / 2.0f);
        int Y = round((double)((x - dst.rows / 2.0f)*sin(angle)) + ((y - dst.cols / 2.0f)*cos(angle)) + src.cols / 2.0f);


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
        if ((distance_x[0] >= 0) && (distance_x[3] < src.rows) && (distance_y[0] >= 0) && (distance_y[3] < src.cols))
        {
          for (int k = 0; k < 4; k++)
          {
            k_i[k] = BiCubicFunction(double_X - distance_x[k]);
            k_j[k] = BiCubicFunction(double_Y - distance_y[k]);
          }
          if (src.channels() == 1)
          {
            double temp = 0;
            if (double_X < 0 || double_X >= src.rows || double_Y < 0 || double_Y >= src.cols)
              temp = 0;
            else
            {
              for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                {
                  temp = temp + (double)src.at<uchar>(distance_x[i], distance_y[j])*k_i[i] * k_j[j];
                }
            }

            dst.at<uchar>(x, y) = (uchar)temp;
          }

          if (src.channels() == 3)
          {
            cv::Vec3f temp = { 0,0,0 };
            if (double_X < 0 || double_X >= src.rows || double_Y < 0 || double_Y >= src.cols)
              temp = { 0,0,0 };
            else
            {
              for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                {
                  temp = temp + (cv::Vec3f)src.at<cv::Vec3b>(distance_x[i], distance_y[j])*k_i[i] * k_j[j];
                }
            }
            dst.at<cv::Vec3b>(x, y) = (cv::Vec3b)temp;
          }
        }
      }
  }
}

void ImageNoise::rotate(cv::Mat &img, int theta, int type, int numofthread)
{
  double angle = (double)theta*PI / 180.0f;
  int height = img.rows;
  int width = img.cols;
  //将坐标中心定义到图像中心，方便旋转
  int imgX1 = -width / 2;//左上顶点
  int imgY1 = height / 2;
  int imgX2 = width / 2;//右上顶点
  int imgY2 = height / 2;
  int imgX3 = -width / 2;//左下顶点
  int imgY3 = -height / 2;
  int imgX4 = width / 2;//右下顶点
  int imgY4 = -height / 2;

  //顺旋转矩阵：[cosθ， -sinθ]
  //         [sinθ, cosθ]
  //加上0.5用于四舍五入
  int dstX1 = (int)(imgX1*cos(angle) - imgY1*sin(angle) + 0.5);
  int dstY1 = (int)(imgX1*sin(angle) + imgY1*cos(angle) + 0.5);
  int dstX2 = (int)(imgX2*cos(angle) - imgY2*sin(angle) + 0.5);
  int dstY2 = (int)(imgX2*sin(angle) + imgY2*cos(angle) + 0.5);
  int dstX3 = (int)(imgX3*cos(angle) - imgY3*sin(angle) + 0.5);
  int dstY3 = (int)(imgX3*sin(angle) + imgY3*cos(angle) + 0.5);
  int dstX4 = (int)(imgX4*cos(angle) - imgY4*sin(angle) + 0.5);
  int dstY4 = (int)(imgX4*sin(angle) + imgY4*cos(angle) + 0.5);

  int dstWidth = cv::max(abs(dstX1 - dstX4), abs(dstX2 - dstX3)) + 1;
  int dstHeight = cv::max(abs(dstY1 - dstY4), abs(dstY2 - dstY3)) + 1;

  cv::Mat dst = cv::Mat::zeros(dstHeight, dstWidth, img.type());
  //imshow("output", dst);
  if (type == 4)
  {
    double * pixel = (double*)malloc(sizeof(double)*width*height);
    double * dst_pixel = (double*)malloc(sizeof(double)*dstHeight*dstWidth);
    double * pixelR = (double*)malloc(sizeof(double)*width*height);
    double * pixelG = (double*)malloc(sizeof(double)*width*height);
    double * pixelB = (double*)malloc(sizeof(double)*width*height);
    double * dst_pixelR = (double*)malloc(sizeof(double)*dstHeight*dstWidth);
    double * dst_pixelG = (double*)malloc(sizeof(double)*dstHeight*dstWidth);
    double * dst_pixelB = (double*)malloc(sizeof(double)*dstHeight*dstWidth);

    if (img.channels() == 1)
    {
      for (int i = 0; i < height; i++)
      {
        for (int j = 0; j < width; j++)
        {
          pixel[i*width + j] = (double)img.at<uchar>(i, j);
        }
      }

      for (int i = 0; i < dstHeight; i++)
      {
        for (int j = 0; j < dstWidth; j++)
        {
          dst_pixel[i*dstWidth + j] = 0;
        }
      }

      rotation_host(pixel, dst_pixel, width, height, dstWidth, dstHeight, angle);

      for (int i = 0; i < dstHeight; i++)
      {
        for (int j = 0; j < dstWidth; j++)
        {
          dst.at<uchar>(i, j) = (uchar)dst_pixel[i*dstWidth + j];
        }
      }
      img = dst.clone();
    }
    else if (img.channels() == 3)
    {
      for (int i = 0; i < height; i++)
      {
        for (int j = 0; j < width; j++)
        {
          pixelR[i*width + j] = (double)img.at<cv::Vec3b>(i, j)[0];
          pixelG[i*width + j] = (double)img.at<cv::Vec3b>(i, j)[1];
          pixelB[i*width + j] = (double)img.at<cv::Vec3b>(i, j)[2];
        }
      }
      
      for (int i = 0; i < dstHeight; i++)
      {
        for (int j = 0; j < dstWidth; j++)
        {
          dst_pixelR[i*dstWidth + j] = 0;
          dst_pixelG[i*dstWidth + j] = 0;
          dst_pixelB[i*dstWidth + j] = 0;
        }
      }

      rotation_host(pixelR, dst_pixelR, width, height, dstWidth, dstHeight, angle);
      rotation_host(pixelG, dst_pixelG, width, height, dstWidth, dstHeight, angle);
      rotation_host(pixelB, dst_pixelB, width, height, dstWidth, dstHeight, angle);

      for (int i = 0; i < dstHeight; i++)
      {
        for (int j = 0; j < dstWidth; j++)
        {
          dst.at<cv::Vec3b>(i, j)[0] = (uchar)dst_pixelR[i*dstWidth + j];
          dst.at<cv::Vec3b>(i, j)[1] = (uchar)dst_pixelG[i*dstWidth + j];
          dst.at<cv::Vec3b>(i, j)[2] = (uchar)dst_pixelB[i*dstWidth + j];
        }
      }

      img = dst.clone();
    }
    delete pixel;
    delete pixelR;
    delete pixelG;
    delete pixelB;
    delete dst_pixel;
    delete dst_pixelR;
    delete dst_pixelG;
    delete dst_pixelB;
  }
  else if (type == 5)
  {
    double * pixel = (double*)malloc(sizeof(double)*width*height);
    double * dst_pixel = (double*)malloc(sizeof(double)*dstHeight*dstWidth);
    double * pixelR = (double*)malloc(sizeof(double)*width*height);
    double * pixelG = (double*)malloc(sizeof(double)*width*height);
    double * pixelB = (double*)malloc(sizeof(double)*width*height);
    double * dst_pixelR = (double*)malloc(sizeof(double)*dstHeight*dstWidth);
    double * dst_pixelG = (double*)malloc(sizeof(double)*dstHeight*dstWidth);
    double * dst_pixelB = (double*)malloc(sizeof(double)*dstHeight*dstWidth);

    if (img.channels() == 1)
    {
      for (int i = 0; i < height; i++)
      {
        for (int j = 0; j < width; j++)
        {
          pixel[i*width + j] = (double)img.at<uchar>(i, j);
        }
      }

      for (int i = 0; i < dstHeight; i++)
      {
        for (int j = 0; j < dstWidth; j++)
        {
          dst_pixel[i*dstWidth + j] = 0;
        }
      }

      rotationCL(pixel, dst_pixel, width, height, dstWidth, dstHeight, angle);

      for (int i = 0; i < dstHeight; i++)
      {
        for (int j = 0; j < dstWidth; j++)
        {
          dst.at<uchar>(i, j) = (uchar)dst_pixel[i*dstWidth + j];
        }
      }
      img = dst.clone();
    }
    else if (img.channels() == 3)
    {
      for (int i = 0; i < height; i++)
      {
        for (int j = 0; j < width; j++)
        {
          pixelR[i*width + j] = (double)img.at<cv::Vec3b>(i, j)[0];
          pixelG[i*width + j] = (double)img.at<cv::Vec3b>(i, j)[1];
          pixelB[i*width + j] = (double)img.at<cv::Vec3b>(i, j)[2];
        }
      }

      for (int i = 0; i < dstHeight; i++)
      {
        for (int j = 0; j < dstWidth; j++)
        {
          dst_pixelR[i*dstWidth + j] = 0;
          dst_pixelG[i*dstWidth + j] = 0;
          dst_pixelB[i*dstWidth + j] = 0;
        }
      }

      rotationCL(pixelR, dst_pixelR, width, height, dstWidth, dstHeight, angle);
      rotationCL(pixelG, dst_pixelG, width, height, dstWidth, dstHeight, angle);
      rotationCL(pixelB, dst_pixelB, width, height, dstWidth, dstHeight, angle);

      for (int i = 0; i < dstHeight; i++)
      {
        for (int j = 0; j < dstWidth; j++)
        {
          dst.at<cv::Vec3b>(i, j)[0] = (uchar)dst_pixelR[i*dstWidth + j];
          dst.at<cv::Vec3b>(i, j)[1] = (uchar)dst_pixelG[i*dstWidth + j];
          dst.at<cv::Vec3b>(i, j)[2] = (uchar)dst_pixelB[i*dstWidth + j];
        }
      }

      img = dst.clone();
    }
    delete pixel;
    delete pixelR;
    delete pixelG;
    delete pixelB;
    delete dst_pixel;
    delete dst_pixelR;
    delete dst_pixelG;
    delete dst_pixelB;
  }
  else
  {
    BicubicInterpolation(img, dst, angle, type, numofthread);
    img = dst.clone();
  } 
  //逆旋转矩阵：[cosθ, sinθ]
  //          [-sinθ, cosθ]
}

void ImageNoise::BiCubicInterpolation_CL(double * cl_src, double * cl_dst, int Width, int Height, int width, int height)
{
  //step 1:get platform;
  cl_int ret;														//errcode;
  cl_uint num_platforms;											//用于保存获取到的platforms数量;
  ret = clGetPlatformIDs(0, NULL, &num_platforms);
  if ((CL_SUCCESS != ret) || (num_platforms < 1))
  {
    cout << "Error getting platform number: " << ret << endl;
    return;
  }
  cl_platform_id platform_id = NULL;
  ret = clGetPlatformIDs(1, &platform_id, NULL);					//获取第一个platform的id;
  if (CL_SUCCESS != ret)
  {
    cout << "Error getting platform id:" << ret;
    return;
  }

  //step 2:get device ;
  cl_uint num_devices;
  ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
  if ((CL_SUCCESS != ret) || (num_devices < 1))
  {
    cout << "Error getting GPU device number:" << ret;
    return;
  }
  cl_device_id device_id = NULL;
  ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
  if (CL_SUCCESS != ret)
  {
    cout << "Error getting GPU device id: " << ret;
    return;
  }

  //step 3:create context;
  cl_context_properties props[] =
  {
    CL_CONTEXT_PLATFORM, (cl_context_properties)platform_id, 0
  };
  cl_context context = NULL;
  context = clCreateContext(props, 1, &device_id, NULL, NULL, &ret);
  if ((CL_SUCCESS != ret) || (NULL == context))
  {
    cout<<"Error createing context: " << ret;
    return;
  }

  //step 4:create command queue;						//一个device有多个queue，queue之间并行执行
  cl_command_queue command_queue = NULL;
  command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
  if ((CL_SUCCESS != ret) || (NULL == command_queue))
  {
    cout << "Error createing command queue: " << ret;
    return;
  }

  //step 5:create memory object;						//缓存类型（buffer），图像类型（iamge）

  cl_mem mem_obj = NULL;
  cl_mem mem_objout = NULL;

  //create opencl memory object using host ptr
  //	mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, BUF_SIZE, host_buffer, &ret);
  mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, Width * Height * sizeof(double), cl_src, &ret);
  mem_objout = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, width * height * sizeof(double), cl_dst, &ret);
  if ((CL_SUCCESS != ret) || (NULL == mem_obj))
  {
    cout << "Error creating memory object: " << ret;
    return;
  }

  //step 6:create program;
  size_t szKernelLength = 0;
  //	const char* oclSourceFile = "add_vector.cl";
  const char* oclSourceFile = "bicubicinterpolation.cl";
  const char* kernelSource = LoadProgSource(oclSourceFile, "", &szKernelLength);
  if (kernelSource == NULL)
  {
    cout << "Error loading source file: " << ret;
    return;
  }

  //create program
  cl_program program = NULL;
  program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, NULL, &ret);
  if ((CL_SUCCESS != ret) || (NULL == program))
  {
    cout << "Error creating program: " << ret;
    return;
  }

  //build program 
  ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
  if (CL_SUCCESS != ret)
  {
    size_t len;
    char buffer[8 * 1024];
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
    cout << (string)buffer << endl;
    return;
  }

  //step 7:create kernel;
  cl_kernel kernel = NULL;
  //	kernel = clCreateKernel(program, "test", &ret);
  kernel = clCreateKernel(program, "bicubicinterpolation_cl", &ret);
  if ((CL_SUCCESS != ret) || (NULL == kernel))
  {
    cout << "Error creating kernel: " << ret;
    return;
  }

  //step 8:set kernel arguement;
  ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)& mem_obj);
  if (CL_SUCCESS != ret)
  {
    cout << "Error setting kernel arguement 0: " << ret;
    return;
  }
  ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)& mem_objout);
  if (CL_SUCCESS != ret)
  {
    cout << "Error setting kernel arguement 1: " << ret;
    return;
  }
  ret = clSetKernelArg(kernel, 2, sizeof(int), (void*)& Width);
  if (CL_SUCCESS != ret)
  {
    cout << "Error setting kernel arguement 2: " << ret;
    return;
  }
  ret = clSetKernelArg(kernel, 3, sizeof(int), (void*)& Height);
  if (CL_SUCCESS != ret)
  {
    cout << "Error setting kernel arguement 3: " << ret;
    return;
  }
  ret = clSetKernelArg(kernel, 4, sizeof(int), (void*)& width);
  if (CL_SUCCESS != ret)
  {
    cout << "Error setting kernel arguement 4: " << ret;
    return;
  }
  ret = clSetKernelArg(kernel, 5, sizeof(int), (void*)& height);
  if (CL_SUCCESS != ret)
  {
    cout << "Error setting kernel arguement 5: " << ret;
    return;
  }
  //step 9:set work group size;  							//<---->dimBlock\dimGrid
  cl_uint work_dim = 2;
  size_t local_work_size[2] = { 32, 32 };
  size_t global_work_size[2] = { RoundUp(local_work_size[0], width),
    RoundUp(local_work_size[1], height) };		//let opencl device determine how to break work items into work groups;

  //step 10:run kernel;				//put kernel and work-item arugement into queue and excute;
  ret = clEnqueueNDRangeKernel(command_queue, kernel, work_dim, NULL, global_work_size, local_work_size, 0, NULL, NULL);
  if (CL_SUCCESS != ret)
  {
    cout << "Error enqueue NDRange: " << ret;
    return;
  }

  //step 11:get result;
  double *device_buffer = (double *)clEnqueueMapBuffer(command_queue, mem_objout, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, width * height * sizeof(double), 0, NULL, NULL, &ret);
  if ((CL_SUCCESS != ret) || (NULL == device_buffer))
  {
    cout << "Error map buffer: " << ret;
    return;
  }

  memcpy(cl_dst, device_buffer, width * height * sizeof(double));
  //step 12:release all resource;
  if (NULL != kernel)
    clReleaseKernel(kernel);
  if (NULL != program)
    clReleaseProgram(program);
  if (NULL != mem_obj)
    clReleaseMemObject(mem_obj);
  if (NULL != command_queue)
    clReleaseCommandQueue(command_queue);
  if (NULL != context)
    clReleaseContext(context);
}

void ImageNoise::rotationCL(double * cl_src, double * cl_dst, int Width, int Height, int width, int height, double angle)
{
  //step 1:get platform;
  cl_int ret;														//errcode;
  cl_uint num_platforms;											//用于保存获取到的platforms数量;
  ret = clGetPlatformIDs(0, NULL, &num_platforms);
  if ((CL_SUCCESS != ret) || (num_platforms < 1))
  {
    cout << "Error getting platform number: " << ret << endl;
    return;
  }
  cl_platform_id platform_id = NULL;
  ret = clGetPlatformIDs(1, &platform_id, NULL);					//获取第一个platform的id;
  if (CL_SUCCESS != ret)
  {
    cout << "Error getting platform id:" << ret;
    return;
  }

  //step 2:get device ;
  cl_uint num_devices;
  ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
  if ((CL_SUCCESS != ret) || (num_devices < 1))
  {
    cout << "Error getting GPU device number:" << ret;
    return;
  }
  cl_device_id device_id = NULL;
  ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
  if (CL_SUCCESS != ret)
  {
    cout << "Error getting GPU device id: " << ret;
    return;
  }

  //step 3:create context;
  cl_context_properties props[] =
  {
    CL_CONTEXT_PLATFORM, (cl_context_properties)platform_id, 0
  };
  cl_context context = NULL;
  context = clCreateContext(props, 1, &device_id, NULL, NULL, &ret);
  if ((CL_SUCCESS != ret) || (NULL == context))
  {
    cout << "Error createing context: " << ret;
    return;
  }

  //step 4:create command queue;						//一个device有多个queue，queue之间并行执行
  cl_command_queue command_queue = NULL;
  command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
  if ((CL_SUCCESS != ret) || (NULL == command_queue))
  {
    cout << "Error createing command queue: " << ret;
    return;
  }

  //step 5:create memory object;						//缓存类型（buffer），图像类型（iamge）

  cl_mem mem_obj = NULL;
  cl_mem mem_objout = NULL;

  //create opencl memory object using host ptr
  //	mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, BUF_SIZE, host_buffer, &ret);
  mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, Width * Height * sizeof(double), cl_src, &ret);
  mem_objout = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, width * height * sizeof(double), cl_dst, &ret);
  if ((CL_SUCCESS != ret) || (NULL == mem_obj))
  {
    cout << "Error creating memory object: " << ret;
    return;
  }

  //step 6:create program;
  size_t szKernelLength = 0;
  //	const char* oclSourceFile = "add_vector.cl";
  const char* oclSourceFile = "rotation.cl";
  const char* kernelSource = LoadProgSource(oclSourceFile, "", &szKernelLength);
  if (kernelSource == NULL)
  {
    cout << "Error loading source file: " << ret;
    return;
  }

  //create program
  cl_program program = NULL;
  program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, NULL, &ret);
  if ((CL_SUCCESS != ret) || (NULL == program))
  {
    cout << "Error creating program: " << ret;
    return;
  }

  //build program 
  ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
  if (CL_SUCCESS != ret)
  {
    size_t len;
    char buffer[8 * 1024];
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
    cout << (string)buffer << endl;
    return;
  }

  //step 7:create kernel;
  cl_kernel kernel = NULL;
  //	kernel = clCreateKernel(program, "test", &ret);
  kernel = clCreateKernel(program, "rotation_cl", &ret);
  if ((CL_SUCCESS != ret) || (NULL == kernel))
  {
    cout << "Error creating kernel: " << ret;
    return;
  }

  //step 8:set kernel arguement;
  ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)& mem_obj);
  if (CL_SUCCESS != ret)
  {
    cout << "Error setting kernel arguement 0: " << ret;
    return;
  }
  ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)& mem_objout);
  if (CL_SUCCESS != ret)
  {
    cout << "Error setting kernel arguement 1: " << ret;
    return;
  }
  ret = clSetKernelArg(kernel, 2, sizeof(int), (void*)& Width);
  if (CL_SUCCESS != ret)
  {
    cout << "Error setting kernel arguement 2: " << ret;
    return;
  }
  ret = clSetKernelArg(kernel, 3, sizeof(int), (void*)& Height);
  if (CL_SUCCESS != ret)
  {
    cout << "Error setting kernel arguement 3: " << ret;
    return;
  }
  ret = clSetKernelArg(kernel, 4, sizeof(int), (void*)& width);
  if (CL_SUCCESS != ret)
  {
    cout << "Error setting kernel arguement 4: " << ret;
    return;
  }
  ret = clSetKernelArg(kernel, 5, sizeof(int), (void*)& height);
  if (CL_SUCCESS != ret)
  {
    cout << "Error setting kernel arguement 5: " << ret;
    return;
  }
  ret = clSetKernelArg(kernel, 6, sizeof(double), (void*)& angle);
  if (CL_SUCCESS != ret)
  {
    cout << "Error setting kernel arguement 6: " << ret;
    return;
  }
  //step 9:set work group size;  							//<---->dimBlock\dimGrid
  cl_uint work_dim = 2;
  size_t local_work_size[2] = { 32, 32 };
  size_t global_work_size[2] = { RoundUp(local_work_size[0], width),
    RoundUp(local_work_size[1], height) };		//let opencl device determine how to break work items into work groups;

                                              //step 10:run kernel;				//put kernel and work-item arugement into queue and excute;
  ret = clEnqueueNDRangeKernel(command_queue, kernel, work_dim, NULL, global_work_size, local_work_size, 0, NULL, NULL);
  if (CL_SUCCESS != ret)
  {
    cout << "Error enqueue NDRange: " << ret;
    return;
  }

  //step 11:get result;
  double *device_buffer = (double *)clEnqueueMapBuffer(command_queue, mem_objout, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, width * height * sizeof(double), 0, NULL, NULL, &ret);
  if ((CL_SUCCESS != ret) || (NULL == device_buffer))
  {
    cout << "Error map buffer: " << ret;
    return;
  }

  memcpy(cl_dst, device_buffer, width * height * sizeof(double));
  //step 12:release all resource;
  if (NULL != kernel)
    clReleaseKernel(kernel);
  if (NULL != program)
    clReleaseProgram(program);
  if (NULL != mem_obj)
    clReleaseMemObject(mem_obj);
  if (NULL != command_queue)
    clReleaseCommandQueue(command_queue);
  if (NULL != context)
    clReleaseContext(context);
}

void ImageNoise::gaussianFilterCL(double * cl_src, double * cl_dst, int width, int height, double * matrix)
{
  //step 1:get platform;
  cl_int ret;														//errcode;
  cl_uint num_platforms;											//用于保存获取到的platforms数量;
  ret = clGetPlatformIDs(0, NULL, &num_platforms);
  if ((CL_SUCCESS != ret) || (num_platforms < 1))
  {
    cout << "Error getting platform number: " << ret << endl;
    return;
  }
  cl_platform_id platform_id = NULL;
  ret = clGetPlatformIDs(1, &platform_id, NULL);					//获取第一个platform的id;
  if (CL_SUCCESS != ret)
  {
    cout << "Error getting platform id:" << ret;
    return;
  }

  //step 2:get device ;
  cl_uint num_devices;
  ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
  if ((CL_SUCCESS != ret) || (num_devices < 1))
  {
    cout << "Error getting GPU device number:" << ret;
    return;
  }
  cl_device_id device_id = NULL;
  ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
  if (CL_SUCCESS != ret)
  {
    cout << "Error getting GPU device id: " << ret;
    return;
  }

  //step 3:create context;
  cl_context_properties props[] =
  {
    CL_CONTEXT_PLATFORM, (cl_context_properties)platform_id, 0
  };
  cl_context context = NULL;
  context = clCreateContext(props, 1, &device_id, NULL, NULL, &ret);
  if ((CL_SUCCESS != ret) || (NULL == context))
  {
    cout << "Error createing context: " << ret;
    return;
  }

  //step 4:create command queue;						//一个device有多个queue，queue之间并行执行
  cl_command_queue command_queue = NULL;
  command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
  if ((CL_SUCCESS != ret) || (NULL == command_queue))
  {
    cout << "Error createing command queue: " << ret;
    return;
  }

  //step 5:create memory object;						//缓存类型（buffer），图像类型（iamge）

  cl_mem mem_obj = NULL;
  cl_mem mem_objout = NULL;
  cl_mem mem_objin = NULL;

  //create opencl memory object using host ptr
  //	mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, BUF_SIZE, host_buffer, &ret);
  mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, width * height * sizeof(double), cl_src, &ret);
  mem_objout = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, width * height * sizeof(double), cl_dst, &ret);
  mem_objin = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, 3 * 3 * sizeof(double), matrix, &ret);
  if ((CL_SUCCESS != ret) || (NULL == mem_obj))
  {
    cout << "Error creating memory object: " << ret;
    return;
  }

  //step 6:create program;
  size_t szKernelLength = 0;
  //	const char* oclSourceFile = "add_vector.cl";
  const char* oclSourceFile = "gaussianFilter.cl";
  const char* kernelSource = LoadProgSource(oclSourceFile, "", &szKernelLength);
  if (kernelSource == NULL)
  {
    cout << "Error loading source file: " << ret;
    return;
  }

  //create program
  cl_program program = NULL;
  program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, NULL, &ret);
  if ((CL_SUCCESS != ret) || (NULL == program))
  {
    cout << "Error creating program: " << ret;
    return;
  }

  //build program 
  ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
  if (CL_SUCCESS != ret)
  {
    size_t len;
    char buffer[8 * 1024];
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
    cout << (string)buffer << endl;
    return;
  }

  //step 7:create kernel;
  cl_kernel kernel = NULL;
  //	kernel = clCreateKernel(program, "test", &ret);
  kernel = clCreateKernel(program, "gaussianFilter_cl", &ret);
  if ((CL_SUCCESS != ret) || (NULL == kernel))
  {
    cout << "Error creating kernel: " << ret;
    return;
  }

  //step 8:set kernel arguement;
  ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)& mem_obj);
  if (CL_SUCCESS != ret)
  {
    cout << "Error setting kernel arguement 0: " << ret;
    return;
  }
  ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)& mem_objout);
  if (CL_SUCCESS != ret)
  {
    cout << "Error setting kernel arguement 1: " << ret;
    return;
  }
  ret = clSetKernelArg(kernel, 2, sizeof(int), (void*)& width);
  if (CL_SUCCESS != ret)
  {
    cout << "Error setting kernel arguement 2: " << ret;
    return;
  }
  ret = clSetKernelArg(kernel, 3, sizeof(int), (void*)& height);
  if (CL_SUCCESS != ret)
  {
    cout << "Error setting kernel arguement 3: " << ret;
    return;
  }
  ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)& mem_objin);
  if (CL_SUCCESS != ret)
  {
    cout << "Error setting kernel arguement 4: " << ret;
    return;
  }
  //step 9:set work group size;  							//<---->dimBlock\dimGrid
  cl_uint work_dim = 2;
  size_t local_work_size[2] = { 32, 32 };
  size_t global_work_size[2] = { RoundUp(local_work_size[0], width) ,
    RoundUp(local_work_size[1], height) };		//let opencl device determine how to break work items into work groups;

                                              //step 10:run kernel;				//put kernel and work-item arugement into queue and excute;
  ret = clEnqueueNDRangeKernel(command_queue, kernel, work_dim, NULL, global_work_size, local_work_size, 0, NULL, NULL);
  if (CL_SUCCESS != ret)
  {
    cout << "Error enqueue NDRange: " << ret;
    return;
  }

  //step 11:get result;
  double *device_buffer = (double *)clEnqueueMapBuffer(command_queue, mem_objout, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, width * height * sizeof(double), 0, NULL, NULL, &ret);
  if ((CL_SUCCESS != ret) || (NULL == device_buffer))
  {
    cout << "Error map buffer: " << ret;
    return;
  }

  memcpy(cl_dst, device_buffer, width * height * sizeof(double));
  //step 12:release all resource;
  if (NULL != kernel)
    clReleaseKernel(kernel);
  if (NULL != program)
    clReleaseProgram(program);
  if (NULL != mem_obj)
    clReleaseMemObject(mem_obj);
  if (NULL != command_queue)
    clReleaseCommandQueue(command_queue);
  if (NULL != context)
    clReleaseContext(context);
}

char * ImageNoise::LoadProgSource(const char * cFilename, const char * cPreamble, size_t * szFinalLength)
{
  FILE* pFileStream = NULL;
  size_t szSourceLength;

  // open the OpenCL source code file  
  pFileStream = fopen(cFilename, "rb");
  if (pFileStream == NULL)
  {
    return NULL;
  }

  size_t szPreambleLength = strlen(cPreamble);

  // get the length of the source code  
  fseek(pFileStream, 0, SEEK_END);
  szSourceLength = ftell(pFileStream);
  fseek(pFileStream, 0, SEEK_SET);

  // allocate a buffer for the source code string and read it in  
  char* cSourceString = (char *)malloc(szSourceLength + szPreambleLength + 1);
  memcpy(cSourceString, cPreamble, szPreambleLength);
  if (fread((cSourceString)+szPreambleLength, szSourceLength, 1, pFileStream) != 1)
  {
    fclose(pFileStream);
    free(cSourceString);
    return 0;
  }

  // close the file and return the total length of the combined (preamble + source) string  
  fclose(pFileStream);
  if (szFinalLength != 0)
  {
    *szFinalLength = szSourceLength + szPreambleLength;
  }
  cSourceString[szSourceLength + szPreambleLength] = '\0';

  return cSourceString;
}

size_t ImageNoise::RoundUp(int groupSize, int globalSize)
{
  int r = globalSize % groupSize;
  if (r == 0)
  {
    return globalSize;
  }
  else
  {
    return globalSize + groupSize - r;
  }
}
