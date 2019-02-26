#include"dft.h"
#include "ui_mainwindow.h"
#include<QDebug>
#include<omp.h>

DFT::DFT()
{

}

myComplex DFT::Mul(myComplex a, myComplex b)
{
  return{ a.real*b.real - a.img*b.img, a.real*b.img + a.img*b.real };
}

myComplex DFT::Add(myComplex a, myComplex b)
{
  return{ a.real + b.real, a.img + b.img };
}

//码位倒置，采用雷德算法
void DFT::reverse(int *data, int length)
{
    int j = length/2;
    int temp;
    for(int i=1; i<length-1; i++)
    {
        if(i<j)
        {
            temp = data[i];
            data[i] = data[j];
            data[j] = temp;
        }
        int k = length/2;
        while(k<=j)
        {
            j -= k;
            k /= 2;
        }
        j += k;
    }
}

//判断是否是2的整数次方
int DFT::if_binaryNum(int length)
{
  int num = 0;

  while (length != 1)
  {
    if (length % 2 == 0)
    {
      length = length / 2;
      num++;
    }
    else
    {
      return -1;
    }
  }

  return num;
}

//将非2的整数次方边长的图片缩放为2的整数次方
void DFT::resizeImage(Mat &image)
{
  float c = image.cols, r = image.rows;
  int cn = 0, rn = 0, cnew = 2, rnew = 2;

  while (c / 2 > 1)
  {
      c = c / 2;
      cn++;
  }
  while (r / 2 > 1)
  {
      r = r / 2;
      rn++;
  }
  while (cn > 0)
  {
      cnew = cnew * 2;
      cn--;
  }
  while (rn > 0)
  {
      rnew = rnew * 2;
      rn--;
  }
  resize(image, image, Size(cnew, rnew));

}

void DFT::fft(Mat &image, int type, int numofthread)
{
  int lengthC = image.cols;
  int lengthR = image.rows;
  int numC, numR;

  myComplex * resultE = new myComplex[lengthC*lengthR];

  //映射表
  int * mappingC = new int[lengthC];
  int * mappingR = new int[lengthR];

  //W值表
  myComplex * mappingWC = new myComplex[lengthC / 2];
  myComplex * mappingWR = new myComplex[lengthR / 2];

  //判断输入图片边长是否是2的n次方，如果不符合，调整image大小
  numC = if_binaryNum(lengthC);
  numR = if_binaryNum(lengthR);

  if (numC == -1 || numR == -1)
  {
    resizeImage(image);
    fft(image, type, numofthread);
    return;
  }

  //构造映射表
  for (int c = 0; c < lengthC; c++)
  {
    mappingC[c] = 0.0f;
  }

  for (int r = 0; r < lengthR; r++)
  {
    mappingR[r] = 0.0f;
  }

  for (int c = 0; c < lengthC; c++)
  {
    mappingC[c] = c;
  }

  for (int r = 0; r < lengthR; r++)
  {
    mappingR[r] = r;
  }

  reverse(mappingC, lengthC);
  reverse(mappingR, lengthR);

  //构造W表
  for (int i = 0; i < lengthC / 2; i++)
  {
    myComplex w = { cosf(2 * PI / lengthC * i), -1 * sinf(2 * PI / lengthC * i) };
    mappingWC[i] = w;
  }

  for (int i = 0; i < lengthR / 2; i++)
  {
    myComplex w = { cosf(2 * PI / lengthR * i), -1 * sinf(2 * PI / lengthR * i) };
    mappingWR[i] = w;
  }

  //初始化
  for (int r = 0; r < lengthR; r++)
  {
    for (int c = 0; c < lengthC; c++)
    {
      //利用映射表，并且以0到1区间的32位浮点类型存储灰度值
      myComplex w = { (float)image.at<uchar>(mappingR[r], mappingC[c]) / 255, 0 };
      resultE[r*lengthC + c] = w;
    }
  }

  //循环计算每行
  if(type = 1)
  {
      for (int r = 0; r < lengthR; r++)
      {
        //循环更新resultE中当前行的数值，即按照蝶形向前层层推进
        for (int i = 0; i < numC; i++)
        {
          int combineSize = 2 << i;
          myComplex * newRow = new myComplex[lengthC];
          //按照2,4,8,16...为单位进行合并，并更新节点的值
          for (int j = 0; j < lengthC; j = j + combineSize)//k+j代表列数
          {
            int n;
            for (int k = 0; k < combineSize; k++)
            {
              //按照蝶形算法的步骤
              if (k < (combineSize / 2))
              {
                int w = k * lengthC / combineSize;//lengthC/combineSize代表合并的DFT层级
                n = k + j + r*lengthC;
                newRow[j+k] = Add(resultE[n], Mul(resultE[n + (combineSize / 2)], mappingWC[w]));
              }
              else
              {
                int w = (k - (combineSize / 2)) * lengthC / combineSize;
                n = k + j - (combineSize / 2) + r*lengthC;
                newRow[j+k] = Add(resultE[n], Mul({ -1, 0 }, Mul(resultE[n + (combineSize / 2)], mappingWC[w])));
              }
            }
          }

          //用newRow来更新resultE中的值
          for (int j = 0; j < lengthC; j++)
          {
            int n = j + r*lengthC;
            resultE[n] = newRow[j];
          }

          delete newRow;
        }
      }

      //循环计算每列
      for (int c = 0; c < lengthC; c++)
      {
        for (int i = 0; i < numR; i++)
        {
          int combineSize = 2 << i;
          myComplex * newColum = new myComplex[lengthR];
          for (int j = 0; j < lengthR; j = j + combineSize)
          {
            int n;
            for (int k = 0; k < combineSize; k++)
            {
              if (k < (combineSize / 2))
              {
                int w = k * lengthR / combineSize;
                n = (j + k) * lengthC + c;
                newColum[j+k] = Add(resultE[n], Mul(resultE[n + (combineSize / 2)*lengthC], mappingWR[w]));
              }
              else
              {
                int w = (k - (combineSize / 2)) * lengthR / combineSize;
                n = (j + k - (combineSize / 2)) * lengthC + c;
                newColum[j+k] = Add(resultE[n], Mul({ -1, 0 }, Mul(resultE[n + (combineSize / 2)*lengthC], mappingWR[w])));
              }
            }
          }

          //用newColum来更新resultE中的值
          for (int j = 0; j < lengthR; j++)
          {
            int n = j*lengthC + c;
            resultE[n] = newColum[j];
          }

          delete newColum;
        }
      }
  }
  else if(type = 3)
  {
      omp_set_num_threads(numofthread);
      #pragma omp parallel for
      for (int r = 0; r < lengthR; r++)
      {
        //循环更新resultE中当前行的数值，即按照蝶形向前层层推进
        for (int i = 0; i < numC; i++)
        {
          int combineSize = 2 << i;
          myComplex * newRow = new myComplex[lengthC];
          //按照2,4,8,16...为单位进行合并，并更新节点的值
          for (int j = 0; j < lengthC; j = j + combineSize)//k+j代表列数
          {
            int n;
            for (int k = 0; k < combineSize; k++)
            {
              //按照蝶形算法的步骤
              if (k < (combineSize / 2))
              {
                int w = k * lengthC / combineSize;//lengthC/combineSize代表合并的DFT层级
                n = k + j + r*lengthC;
                newRow[j+k] = Add(resultE[n], Mul(resultE[n + (combineSize / 2)], mappingWC[w]));
              }
              else
              {
                int w = (k - (combineSize / 2)) * lengthC / combineSize;
                n = k + j - (combineSize / 2) + r*lengthC;
                newRow[j+k] = Add(resultE[n], Mul({ -1, 0 }, Mul(resultE[n + (combineSize / 2)], mappingWC[w])));
              }
            }
          }

          //用newRow来更新resultE中的值
          for (int j = 0; j < lengthC; j++)
          {
            int n = j + r*lengthC;
            resultE[n] = newRow[j];
          }

          delete newRow;
        }
      }

      //循环计算每列
      omp_set_num_threads(numofthread);
      #pragma omp parallel for
      for (int c = 0; c < lengthC; c++)
      {
        for (int i = 0; i < numR; i++)
        {
          int combineSize = 2 << i;
          myComplex * newColum = new myComplex[lengthR];
          for (int j = 0; j < lengthR; j = j + combineSize)
          {
            int n;
            for (int k = 0; k < combineSize; k++)
            {
              if (k < (combineSize / 2))
              {
                int w = k * lengthR / combineSize;
                n = (j + k) * lengthC + c;
                newColum[j+k] = Add(resultE[n], Mul(resultE[n + (combineSize / 2)*lengthC], mappingWR[w]));
              }
              else
              {
                int w = (k - (combineSize / 2)) * lengthR / combineSize;
                n = (j + k - (combineSize / 2)) * lengthC + c;
                newColum[j+k] = Add(resultE[n], Mul({ -1, 0 }, Mul(resultE[n + (combineSize / 2)*lengthC], mappingWR[w])));
              }
            }
          }

          //用newColum来更新resultE中的值
          for (int j = 0; j < lengthR; j++)
          {
            int n = j*lengthC + c;
            resultE[n] = newColum[j];
          }

          delete newColum;
        }
      }
  }


  //结果存入一个数组中
  float val_max, val_min;
  float * amplitude = new float[lengthC*lengthR];

  for (int r = 0; r < lengthR; r++)
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

  //将数据转存到Mat中，并归一化到0到255区间
  //归一化：(value - min) / (max - min) * 255
  //定义scale = 255 / (max - min)
  Mat fftResult = Mat(lengthC, lengthR, CV_8UC1);
  float scale = 255 / (val_max - val_min);

  for (int i = 0; i < lengthR; i++)
  {
    for (int j = 0; j < lengthC; j++)
    {
      int val = (int)((amplitude[i*lengthC + j] - val_min) * scale);
      fftResult.at<uchar>(i, j) = val;
    }
  }


  //调整象限
  int cx = fftResult.cols / 2;
  int cy = fftResult.rows / 2;

  Mat q0(fftResult, Rect(0, 0, cx, cy));
  Mat q1(fftResult, Rect(cx, 0, cx, cy));
  Mat q2(fftResult, Rect(0, cy, cx, cy));
  Mat q3(fftResult, Rect(cx, cy, cx, cy));

  Mat tmp;

  q0.copyTo(tmp);
  q3.copyTo(q0);
  tmp.copyTo(q3);

  q1.copyTo(tmp);
  q2.copyTo(q1);
  tmp.copyTo(q2);

  image = fftResult.clone();
  //imshow("fft", image);
}
