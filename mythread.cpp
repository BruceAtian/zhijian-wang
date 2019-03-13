#include"mythread.h"
#include"imagenoise.h"

MyThread::MyThread(QThread *parent) :QThread(parent)
{
  isStop = false;
}


void MyThread::closeThread()
{
  isStop = true;
}


void MyThread::startThread_noise(cv::Mat &image, int type, double means, double sigma, double SNR, int k)
{
  ImageNoise imagenoise;

  switch (type) {
  case 1:
    imagenoise.addGaussianNoise(image, k, means, sigma, type);
    break;
  case 2:
    imagenoise.addSaultandPepperNoise(image, SNR);
    break;
  }
  this->quit();
}

void MyThread::startThread_filter(cv::Mat &image, int type, int kernelsize, float filter_sigma, int median_kernelsize, int median_maxsize)
{
  ImageNoise imagenoise;

  switch (type) {
  case 1:
    imagenoise.smoothLinearFilter(image);
    break;
  case 2:
    imagenoise.gaussianFilter(image, filter_sigma, type);
    break;
  case 3:
    imagenoise.weinerFilter(image, kernelsize);
    break;
  case 4:
    imagenoise.medianFilterResult(image, median_kernelsize, median_maxsize);
    break;
  }
  this->quit();
}

void MyThread::startThread_rotate(cv::Mat & src, cv::Mat & dst, double angle, int numofthread, int blocknumber)
{
  ImageNoise imagenoise;
  int width = src.cols;
  int height = src.rows;

  for (int x = 0; x < dst.rows; x++)
    for (int y = dst.cols*blocknumber / numofthread; y < dst.cols*(blocknumber + 1) / numofthread; y++)
    {
      double double_X = ((x - dst.rows / 2.0f)*cos(angle)) - ((y - dst.cols / 2.0f)*sin(angle)) + height / 2.0f;
      double double_Y = ((x - dst.rows / 2.0f)*sin(angle)) + ((y - dst.cols / 2.0f)*cos(angle)) + width / 2.0f;

      int X = round((double)((x - dst.rows / 2.0f)*cos(angle)) - ((y - dst.cols / 2.0f)*sin(angle)) + height / 2.0f);
      int Y = round((double)((x - dst.rows / 2.0f)*sin(angle)) + ((y - dst.cols / 2.0f)*cos(angle)) + width / 2.0f);


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
      if ((distance_x[0] >= 0) && (distance_x[3] < height) && (distance_y[0] >= 0) && (distance_y[3] < width))
      {
        for (int k = 0; k < 4; k++)
        {
          k_i[k] = imagenoise.BiCubicFunction(double_X - distance_x[k]);
          k_j[k] = imagenoise.BiCubicFunction(double_Y - distance_y[k]);
        }
        if (src.channels() == 1)
        {
          double temp = 0;
          if (double_X < 0 || double_X >= height || double_Y < 0 || double_Y >= width)
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
          if (double_X < 0 || double_X >= height || double_Y < 0 || double_Y >= width)
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

void MyThread::startThread_scale(cv::Mat & src, cv::Mat & dst, int numofthread, int blocknumber, double fx, double fy)
{
  ImageNoise imagenoise;
  for (int x = dst.rows*blocknumber / numofthread; x < dst.rows*(blocknumber + 1) / numofthread; x++)
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
          k_i[k] = imagenoise.BiCubicFunction(x / fx - distance_x[k]);
          k_j[k] = imagenoise.BiCubicFunction(y / fy - distance_y[k]);

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

void MyThread::startThread_amplitude(int lengthR, int lengthC, myComplex *resultE, float &val_max, float &val_min, float *amplitude, int numofthread, int blocknumber)
{


  for (int r = lengthR*blocknumber/numofthread; r < lengthR*(blocknumber+1)/numofthread; r++)
  {
    for (int c = 0; c < lengthC; c++)
    {
      myComplex e = resultE[r*lengthC + c];
      float val = sqrt(e.real*e.real + e.img*e.img) + 1;//得到magnitude

      //对数尺度缩放,Log(Mag)
      val = log(val);
      amplitude[r*lengthC + c] = val;

      if (c == 0 && r == 0)
      {
        val_max = val;
        val_min = val;
      }
      else
      {
        if (val_max < val) val_max = val;
        if (val_min > val) val_min = val;
      }
    }
  }
}

void MyThread::startThread_normalize(float &val_max, float &val_min, int lengthR, int lengthC, float *amplitude, cv::Mat &fftResult, int numofthread, int blocknumber)
{
  float scale = 255 / (val_max - val_min);

  for (int i = lengthR*blocknumber/numofthread; i < lengthR*(blocknumber+1)/numofthread; i++)
  {
    for (int j = 0; j < lengthC; j++)
    {
      int val = (int)((amplitude[i*lengthC + j] - val_min) * scale);
      fftResult.at<uchar>(i, j) = val;
    }
  }
}



