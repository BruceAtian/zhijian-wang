#ifndef MYTHREAD_H
#define MYTHREAD_H

#include<QThread>
#include<QObject>
#include<QDebug>
#include<QMessageBox>
#include "dft.h"
#include"imagenoise.h"

class MyThread : public QThread
{
  Q_OBJECT
public:
  explicit MyThread(QThread *parent = nullptr);
  void startThread_noise(cv::Mat &image, int type, double means, double sigma, double SNR, int k);
  void startThread_filter(cv::Mat &image, int type, int kernelsize, float filter_sigma, int median_kernelsize, int median_maxsize);
  void startThread_rotate(cv::Mat &src, cv::Mat &dst, double angle, int numofthread, int blocknumber);
  void startThread_scale(cv::Mat &src, cv::Mat &dst, int numofthread, int blocknumber, double fx, double fy);
  void startThread_amplitude(int lengthR, int lengthC, myComplex *resultE, float &val_max, float &val_min, float *amplitude, int numofthread, int blocknumber);
  void startThread_normalize(float &val_max, float &val_min, int lengthR, int lengthC, float *amplitude, cv::Mat &fftResult, int numofthread, int blocknumber);
  void closeThread();

signals:

  public slots :

private:
  volatile bool isStop;
};

#endif // MYTHREAD_H
