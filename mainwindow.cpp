#include "mainwindow.h"
#include "ui_mainwindow.h"


MainWindow::MainWindow(QWidget *parent) :
  QMainWindow(parent),
  ui(new Ui::MainWindow)
{
  ui->setupUi(this);

  ui->spinBox->setMinimum(1);  // 最小值
  ui->spinBox->setMaximum(8);  // 最大值
  ui->spinBox->setSingleStep(1);  // 步长

  // 滑动条
  ui->linearSlider->setMinimum(1);  // 最小值
  ui->linearSlider->setMaximum(8);  // 最大值
  ui->linearSlider->setSingleStep(1);  // 步长

  connect(ui->comboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(on_comboBox_currentIndexChanged(int)));
  connect(ui->comboBox_3, SIGNAL(currentIndexChanged(int)), this, SLOT(on_comboBox_3_currentIndexChanged(int)));
  connect(ui->spinBox, SIGNAL(valueChanged(int)), ui->linearSlider, SLOT(setValue(int)));
  connect(ui->linearSlider, SIGNAL(valueChanged(int)), ui->spinBox, SLOT(setValue(int)));
}

MainWindow::~MainWindow()
{
  delete ui;
}

void MainWindow::on_pushButton_clicked()
{
  QString fileName;
  QString newFileName = "F:\\temp1.jpg";
  fileName = QFileDialog::getOpenFileName(this, "选择图片文件", "", "Image(*.jpg *.tif *.png *.bmp)");
  if (fileName.isEmpty())
  {
    QMessageBox mesg;
    mesg.warning(this, "警告!", "打开图片失败!");
  }

  QPixmap *img = new QPixmap(fileName);
  ui->label->resize(ui->widget->size());
  img->scaled(ui->label->size(), Qt::KeepAspectRatio);
  ui->label->setScaledContents(true);
  ui->label->setPixmap(*img);

  cv::Mat testImage = cv::imread(fileName.toStdString(), 1);
  cv::imwrite(newFileName.toStdString(), testImage);
}


void MainWindow::on_pushButton_2_clicked()
{
  ImageNoise imgnoise;
  QString fileName = "F:\\temp2.jpg";
  cv::Mat image = cv::imread(fileName.toStdString(), 1);
  int flag = 1;
  int i;
  if (ui->checkBox->isChecked() == true)
    flag = 100;
  int theta = 0, kernelsize = 0;
  float filter_sigma = 0;
  int median_kernelsize = 0, median_maxsize = 0;
  double running_time = 0;

  theta = ui->rotateAngle->text().toInt();
  kernelsize = ui->kernel_size->text().toInt();

  filter_sigma = ui->filter_sigma->text().toFloat();
  median_kernelsize = ui->median_kernelsize->text().toInt();
  median_maxsize = ui->median_maxsize->text().toInt();

  for (i = 0; i < flag; i++)
  {
    if (ui->comboBox->currentIndex() == 1)
    {
      if (ui->comboBox_2->currentIndex() == 1)
      {
        ui->widget_2->show();
        DWORD start_time = GetTickCount();
        imgnoise.smoothLinearFilter(image);
        DWORD end_time = GetTickCount();
        if(flag == 1)
          running_time = (end_time - start_time) / 1000.0;
        else
          running_time += (end_time - start_time) / 1000.0;
        cv::imwrite(fileName.toStdString(), image);
        QPixmap *img_2 = new QPixmap(fileName);
        ui->label_2->resize(ui->widget_2->size());
        img_2->scaled(ui->label_2->size(), Qt::KeepAspectRatio);
        ui->label_2->setScaledContents(true);
        ui->label_2->setPixmap(*img_2);
        ui->textEdit->append(tr("Single thread: Smooth linear filter takes %1s").arg(running_time));
      }

      if (ui->comboBox_2->currentIndex() == 2)
      {
        if (filter_sigma == 0)
        {
          QMessageBox mesg;
          mesg.warning(this, "警告", "请输入参数！");
        }
        ui->widget_2->show();
        DWORD start_time = GetTickCount();
        imgnoise.gaussianFilter(image, filter_sigma, ui->comboBox->currentIndex());
        DWORD end_time = GetTickCount();
        if (flag == 1)
          running_time = (end_time - start_time) / 1000.0;
        else
          running_time += (end_time - start_time) / 1000.0;
        cv::imwrite(fileName.toStdString(), image);
        QPixmap *img_2 = new QPixmap(fileName);
        ui->label_2->resize(ui->widget_2->size());
        img_2->scaled(ui->label_2->size(), Qt::KeepAspectRatio);
        ui->label_2->setScaledContents(true);
        ui->label_2->setPixmap(*img_2);
        ui->textEdit->append(tr("Single thread: Gaussian filter takes %1s").arg(running_time));
      }

      if (ui->comboBox_2->currentIndex() == 3)
      {
        if (kernelsize == 0)
        {
          QMessageBox mesg;
          mesg.warning(this, "警告", "请输入参数！");
        }
        ui->widget_2->show();
        DWORD start_time = GetTickCount();
        imgnoise.weinerFilter(image, kernelsize);
        DWORD end_time = GetTickCount();
        if (flag == 1)
          running_time = (end_time - start_time) / 1000.0;
        else
          running_time += (end_time - start_time) / 1000.0;
        cv::imwrite(fileName.toStdString(), image);
        QPixmap *img_2 = new QPixmap(fileName);
        ui->label_2->resize(ui->widget_2->size());
        img_2->scaled(ui->label_2->size(), Qt::KeepAspectRatio);
        ui->label_2->setScaledContents(true);
        ui->label_2->setPixmap(*img_2);
        ui->textEdit->append(tr("Single thread: Weiner filter takes %1s").arg(running_time));
      }

      if (ui->comboBox_2->currentIndex() == 4)
      {
        if (median_kernelsize == 0 || median_maxsize == 0)
        {
          QMessageBox mesg;
          mesg.warning(this, "警告", "请输入参数！");
        }
        ui->widget_2->show();
        DWORD start_time = GetTickCount();
        imgnoise.medianFilterResult(image, median_kernelsize, median_maxsize);
        DWORD end_time = GetTickCount();
        if (flag == 1)
          running_time = (end_time - start_time) / 1000.0;
        else
          running_time += (end_time - start_time) / 1000.0;
        cv::imwrite(fileName.toStdString(), image);
        QPixmap *img_2 = new QPixmap(fileName);
        ui->label_2->resize(ui->widget_2->size());
        img_2->scaled(ui->label_2->size(), Qt::KeepAspectRatio);
        ui->label_2->setScaledContents(true);
        ui->label_2->setPixmap(*img_2);
        ui->textEdit->append(tr("Single thread: Adaptive Median filter takes %1s").arg(running_time));
      }

      if (ui->comboBox_2->currentIndex() == 5)
      {
        int type = ui->comboBox->currentIndex();
        if (theta == 0)
        {
          QMessageBox mesg;
          mesg.warning(this, "警告", "请输入参数！");
        }
        QString fileName_2 = "F:\\temp1.jpg";
        cv::Mat image_2 = cv::imread(fileName_2.toStdString(), 1);
        ui->widget_2->show();
        DWORD start_time = GetTickCount();
        imgnoise.rotate(image_2, theta, type, ui->spinBox->value());
        DWORD end_time = GetTickCount();
        if (flag == 1)
          running_time = (end_time - start_time) / 1000.0;
        else
          running_time += (end_time - start_time) / 1000.0;
        cv::imwrite(fileName_2.toStdString(), image_2);
        cv::imwrite(fileName.toStdString(), image_2);
        QPixmap *img_2 = new QPixmap(fileName);
        ui->label_2->resize(ui->widget_2->size());
        img_2->scaled(ui->label_2->size(), Qt::KeepAspectRatio);
        ui->label_2->setScaledContents(true);
        ui->label_2->setPixmap(*img_2);
        if(type == 1)
          ui->textEdit->append(tr("Single thread: Rotate takes %1s").arg(running_time));
        else if(type == 3)
          ui->textEdit->append(tr("OpenMP: Rotate takes %1s").arg(running_time));
      }

      if (ui->comboBox_2->currentIndex() == 6)
      {
        DFT dft_obj;
        QString fileName_2 = "F:\\temp1.jpg";
        cv::Mat image_2 = cv::imread(fileName_2.toStdString(), 0);
        ui->widget_2->show();
        DWORD start_time = GetTickCount();
        dft_obj.fft(image_2, 1, 1);
        DWORD end_time = GetTickCount();
        if (flag == 1)
          running_time = (end_time - start_time) / 1000.0;
        else
          running_time += (end_time - start_time) / 1000.0;
        cv::imwrite(fileName.toStdString(), image_2);
        QPixmap *img_2 = new QPixmap(fileName);
        ui->label_2->resize(ui->widget_2->size());
        img_2->scaled(ui->label_2->size(), Qt::KeepAspectRatio);
        ui->label_2->setScaledContents(true);
        ui->label_2->setPixmap(*img_2);
        ui->textEdit->append(tr("Single thread: FFT takes %1s").arg(running_time));
      }
    }
    else if (ui->comboBox->currentIndex() == 2)
    {
      int numofthread = ui->spinBox->value();
      MyThread *mythread = new MyThread[numofthread]();
      cv::Mat *blockImage = new cv::Mat[numofthread];
      DWORD start_time = GetTickCount();
      if (ui->comboBox_2->currentIndex() == 5)
      {
        QString fileName_2 = "F:\\temp1.jpg";
        cv::Mat image_2 = cv::imread(fileName_2.toStdString(), 1);
        DWORD start_time = GetTickCount();
        double angle = (double)theta*PI / 180.0f;
        int height = image_2.rows;
        int width = image_2.cols;
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

        int dstWidth = max(abs(dstX1 - dstX4), abs(dstX2 - dstX3)) + 1;
        int dstHeight = max(abs(dstY1 - dstY4), abs(dstY2 - dstY3)) + 1;

        cv::Mat dst = cv::Mat::zeros(dstHeight, dstWidth, image_2.type());

        for (int i = 0; i < numofthread; i++)
        {
          mythread[i].start();
          mythread[i].startThread_rotate(image_2, dst, angle, numofthread, i);
        }
        DWORD end_time = GetTickCount();
        if (flag == 1)
          running_time = (end_time - start_time) / 1000.0;
        else
          running_time += (end_time - start_time) / 1000.0;
        cv::imwrite(fileName_2.toStdString(), dst);
        cv::imwrite(fileName.toStdString(), dst);
        QPixmap *img_2 = new QPixmap(fileName);
        ui->label_2->resize(ui->widget_2->size());
        img_2->scaled(ui->label_2->size(), Qt::KeepAspectRatio);
        ui->label_2->setScaledContents(true);
        ui->label_2->setPixmap(*img_2);
        ui->textEdit->append(tr("Multi thread: Rotate takes %1s").arg(running_time));
      }
      else if (ui->comboBox_2->currentIndex() == 6)
      {
        DFT dft_obj;
        QString fileName_2 = "F:\\temp1.jpg";
        cv::Mat image_2 = cv::imread(fileName_2.toStdString(), 0);
        DWORD start_time = GetTickCount();

        dft_obj.resizeImage(image_2);

        int lengthC = image_2.cols;
        int lengthR = image_2.rows;
        int numC, numR;

        myComplex * resultE = new myComplex[lengthC*lengthR];

        //映射表
        int * mappingC = new int[lengthC];
        int * mappingR = new int[lengthR];

        //W值表
        myComplex * mappingWC = new myComplex[lengthC / 2];
        myComplex * mappingWR = new myComplex[lengthR / 2];

        numC = dft_obj.if_binaryNum(lengthC);
        numR = dft_obj.if_binaryNum(lengthR);
       
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

        dft_obj.reverse(mappingC, lengthC);
        dft_obj.reverse(mappingR, lengthR);

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
            myComplex w = { (float)image_2.at<uchar>(mappingR[r], mappingC[c]) / 255, 0 };
            resultE[r*lengthC + c] = w;
          }
        }

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
                  newRow[j + k] = dft_obj.Add(resultE[n], dft_obj.Mul(resultE[n + (combineSize / 2)], mappingWC[w]));
                }
                else
                {
                  int w = (k - (combineSize / 2)) * lengthC / combineSize;
                  n = k + j - (combineSize / 2) + r*lengthC;
                  newRow[j + k] = dft_obj.Add(resultE[n], dft_obj.Mul({ -1, 0 }, dft_obj.Mul(resultE[n + (combineSize / 2)], mappingWC[w])));
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
                  newColum[j + k] = dft_obj.Add(resultE[n], dft_obj.Mul(resultE[n + (combineSize / 2)*lengthC], mappingWR[w]));
                }
                else
                {
                  int w = (k - (combineSize / 2)) * lengthR / combineSize;
                  n = (j + k - (combineSize / 2)) * lengthC + c;
                  newColum[j + k] = dft_obj.Add(resultE[n], dft_obj.Mul({ -1, 0 }, dft_obj.Mul(resultE[n + (combineSize / 2)*lengthC], mappingWR[w])));
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

        float val_max, val_min;
        float * amplitude = new float[lengthC*lengthR];

        for (int i = 0; i < numofthread; i++)
        {
          mythread[i].start();
          mythread[i].startThread_amplitude(lengthR, lengthC, resultE, val_max, val_min, amplitude, numofthread, i);
        }

        //将数据转存到Mat中，并归一化到0到255区间
        //归一化：(value - min) / (max - min) * 255
        //定义scale = 255 / (max - min)
        Mat fftResult = Mat(lengthC, lengthR, CV_8UC1);
        for (int i = 0; i < numofthread; i++)
        {
          mythread[i].start();
          mythread[i].startThread_normalize(val_max, val_min, lengthR, lengthC, amplitude, fftResult, numofthread, i);
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

        image_2 = fftResult.clone();
        DWORD end_time = GetTickCount();
        if (flag == 1)
          running_time = (end_time - start_time) / 1000.0;
        else
          running_time += (end_time - start_time) / 1000.0;
        cv::imwrite(fileName.toStdString(), image_2);
        QPixmap *img_2 = new QPixmap(fileName);
        ui->label_2->resize(ui->widget_2->size());
        img_2->scaled(ui->label_2->size(), Qt::KeepAspectRatio);
        ui->label_2->setScaledContents(true);
        ui->label_2->setPixmap(*img_2);
        ui->textEdit->append(tr("Multi thread: FFT takes %1s").arg(running_time));
      }
      else
      {
        for (int i = 0; i < numofthread; i++)
        {
          blockImage[i] = cv::Mat::zeros(image.rows / numofthread, image.cols, image.type());
          for (int x = image.rows / numofthread*i; x < image.rows / numofthread*(i + 1); x++)
            for (int y = 0; y < image.cols; y++)
            {
              if (image.channels() == 3)
                blockImage[i].at<cv::Vec3b>(x - (int)image.rows / numofthread*i, y) = image.at<cv::Vec3b>(x, y);
              else
                blockImage[i].at<uchar>(x - (int)image.rows / numofthread*i, y) = image.at<uchar>(x, y);
            }
        }

        for (int i = 0; i < numofthread; i++)
        {
          mythread[i].start();
          mythread[i].startThread_filter(blockImage[i], ui->comboBox_2->currentIndex(), kernelsize, filter_sigma, median_kernelsize, median_maxsize);
        }

        for (int i = 0; i < numofthread; i++)
        {
          for (int x = image.rows / numofthread*i; x < image.rows / numofthread*(i + 1); x++)
            for (int y = 0; y < image.cols; y++)
            {
              if (image.channels() == 3)
                image.at<cv::Vec3b>(x, y) = blockImage[i].at<cv::Vec3b>(x - (int)image.rows / numofthread*i, y);
              else
                image.at<uchar>(x, y) = blockImage[i].at<uchar>(x - (int)image.rows / numofthread*i, y);
            }
        }
        DWORD end_time = GetTickCount();
        if (flag == 1)
          running_time = (end_time - start_time) / 1000.0;
        else
          running_time += (end_time - start_time) / 1000.0;

        ui->widget_2->show();
        cv::imwrite(fileName.toStdString(), image);
        QPixmap *img_2 = new QPixmap(fileName);
        ui->label_2->resize(ui->widget_2->size());
        img_2->scaled(ui->label_2->size(), Qt::KeepAspectRatio);
        ui->label_2->setScaledContents(true);
        ui->label_2->setPixmap(*img_2);
        if (ui->comboBox_2->currentIndex() == 1)
          ui->textEdit->append(tr("Multi thread: Smooth linear filter takes %1s").arg(running_time));
        else if (ui->comboBox_2->currentIndex() == 2)
          ui->textEdit->append(tr("Multi thread: Gaussian filter takes %1s").arg(running_time));
        else if (ui->comboBox_2->currentIndex() == 3)
          ui->textEdit->append(tr("Multi thread: Weiner filter takes %1s").arg(running_time));
        else if (ui->comboBox_2->currentIndex() == 4)
          ui->textEdit->append(tr("Multi thread: Adaptive median filter takes %1s").arg(running_time));
      }
    }
    else if (ui->comboBox->currentIndex() == 3)
    {
      if (ui->comboBox_2->currentIndex() == 1)
      {
        int numofthread = ui->spinBox->value();
        cv::Mat *blockImage = new cv::Mat[numofthread];
        DWORD start_time = GetTickCount();

        omp_set_num_threads(numofthread);
#pragma omp parallel for
        for (int i = 0; i < numofthread; i++)
        {
          blockImage[i] = cv::Mat::zeros(image.rows / numofthread, image.cols, image.type());
          for (int x = image.rows / numofthread*i; x < image.rows / numofthread*(i + 1); x++)
            for (int y = 0; y < image.cols; y++)
            {
              if (image.channels() == 3)
                blockImage[i].at<cv::Vec3b>(x - (int)image.rows / numofthread*i, y) = image.at<cv::Vec3b>(x, y);
              else
                blockImage[i].at<uchar>(x - (int)image.rows / numofthread*i, y) = image.at<uchar>(x, y);
            }
          imgnoise.smoothLinearFilter(blockImage[i]);
        }

        for (int i = 0; i < numofthread; i++)
        {
          for (int x = image.rows / numofthread*i; x < image.rows / numofthread*(i + 1); x++)
            for (int y = 0; y < image.cols; y++)
            {
              if (image.channels() == 3)
                image.at<cv::Vec3b>(x, y) = blockImage[i].at<cv::Vec3b>(x - (int)image.rows / numofthread*i, y);
              else
                image.at<uchar>(x, y) = blockImage[i].at<uchar>(x - (int)image.rows / numofthread*i, y);
            }
        }
        DWORD end_time = GetTickCount();
        if (flag == 1)
          running_time = (end_time - start_time) / 1000.0;
        else
          running_time += (end_time - start_time) / 1000.0;

        ui->widget_2->show();
        cv::imwrite(fileName.toStdString(), image);
        QPixmap *img_2 = new QPixmap(fileName);
        ui->label_2->resize(ui->widget_2->size());
        img_2->scaled(ui->label_2->size(), Qt::KeepAspectRatio);
        ui->label_2->setScaledContents(true);
        ui->label_2->setPixmap(*img_2);
        ui->textEdit->append(tr("OpenMP: Smooth linear filter takes %1s").arg(running_time));
      }

      if (ui->comboBox_2->currentIndex() == 2)
      {
        if (filter_sigma == 0)
        {
          QMessageBox mesg;
          mesg.warning(this, "警告", "请输入参数！");
        }
        int numofthread = ui->spinBox->value();
        cv::Mat *blockImage = new cv::Mat[numofthread];
        DWORD start_time = GetTickCount();

        omp_set_num_threads(numofthread);
#pragma omp parallel for
        for (int i = 0; i < numofthread; i++)
        {
          blockImage[i] = cv::Mat::zeros(image.rows / numofthread, image.cols, image.type());
          for (int x = image.rows / numofthread*i; x < image.rows / numofthread*(i + 1); x++)
            for (int y = 0; y < image.cols; y++)
            {
              if (image.channels() == 3)
                blockImage[i].at<cv::Vec3b>(x - (int)image.rows / numofthread*i, y) = image.at<cv::Vec3b>(x, y);
              else
                blockImage[i].at<uchar>(x - (int)image.rows / numofthread*i, y) = image.at<uchar>(x, y);
            }
          imgnoise.gaussianFilter(blockImage[i], filter_sigma, ui->comboBox->currentIndex());
        }

        for (int i = 0; i < numofthread; i++)
        {
          for (int x = image.rows / numofthread*i; x < image.rows / numofthread*(i + 1); x++)
            for (int y = 0; y < image.cols; y++)
            {
              if (image.channels() == 3)
                image.at<cv::Vec3b>(x, y) = blockImage[i].at<cv::Vec3b>(x - (int)image.rows / numofthread*i, y);
              else
                image.at<uchar>(x, y) = blockImage[i].at<uchar>(x - (int)image.rows / numofthread*i, y);
            }
        }
        DWORD end_time = GetTickCount();
        if (flag == 1)
          running_time = (end_time - start_time) / 1000.0;
        else
          running_time += (end_time - start_time) / 1000.0;

        ui->widget_2->show();
        cv::imwrite(fileName.toStdString(), image);
        QPixmap *img_2 = new QPixmap(fileName);
        ui->label_2->resize(ui->widget_2->size());
        img_2->scaled(ui->label_2->size(), Qt::KeepAspectRatio);
        ui->label_2->setScaledContents(true);
        ui->label_2->setPixmap(*img_2);
        ui->textEdit->append(tr("OpenMP: Gaussian filter takes %1s").arg(running_time));
      }

      if (ui->comboBox_2->currentIndex() == 3)
      {
        if (kernelsize == 0)
        {
          QMessageBox mesg;
          mesg.warning(this, "警告", "请输入参数！");
        }
        int numofthread = ui->spinBox->value();
        cv::Mat *blockImage = new cv::Mat[numofthread];
        DWORD start_time = GetTickCount();
        omp_set_num_threads(numofthread);
#pragma omp parallel for
        for (int i = 0; i < numofthread; i++)
        {
          blockImage[i] = cv::Mat::zeros(image.rows / numofthread, image.cols, image.type());
          for (int x = image.rows / numofthread*i; x < image.rows / numofthread*(i + 1); x++)
            for (int y = 0; y < image.cols; y++)
            {
              if (image.channels() == 3)
                blockImage[i].at<cv::Vec3b>(x - (int)image.rows / numofthread*i, y) = image.at<cv::Vec3b>(x, y);
              else
                blockImage[i].at<uchar>(x - (int)image.rows / numofthread*i, y) = image.at<uchar>(x, y);
            }
          imgnoise.weinerFilter(blockImage[i], kernelsize);
        }

        for (int i = 0; i < numofthread; i++)
        {
          for (int x = image.rows / numofthread*i; x < image.rows / numofthread*(i + 1); x++)
            for (int y = 0; y < image.cols; y++)
            {
              if (image.channels() == 3)
                image.at<cv::Vec3b>(x, y) = blockImage[i].at<cv::Vec3b>(x - (int)image.rows / numofthread*i, y);
              else
                image.at<uchar>(x, y) = blockImage[i].at<uchar>(x - (int)image.rows / numofthread*i, y);
            }
        }
        DWORD end_time = GetTickCount();
        if (flag == 1)
          running_time = (end_time - start_time) / 1000.0;
        else
          running_time += (end_time - start_time) / 1000.0;

        ui->widget_2->show();
        cv::imwrite(fileName.toStdString(), image);
        QPixmap *img_2 = new QPixmap(fileName);
        ui->label_2->resize(ui->widget_2->size());
        img_2->scaled(ui->label_2->size(), Qt::KeepAspectRatio);
        ui->label_2->setScaledContents(true);
        ui->label_2->setPixmap(*img_2);
        ui->textEdit->append(tr("OpenMP: Weiner filter takes %1s").arg(running_time));
      }

      if (ui->comboBox_2->currentIndex() == 4)
      {
        if (median_kernelsize == 0 || median_maxsize == 0)
        {
          QMessageBox mesg;
          mesg.warning(this, "警告", "请输入参数！");
        }
        int numofthread = ui->spinBox->value();
        cv::Mat *blockImage = new cv::Mat[numofthread];
        DWORD start_time = GetTickCount();
        omp_set_num_threads(numofthread);
#pragma omp parallel for
        for (int i = 0; i < numofthread; i++)
        {
          blockImage[i] = cv::Mat::zeros(image.rows / numofthread, image.cols, image.type());
          for (int x = image.rows / numofthread*i; x < image.rows / numofthread*(i + 1); x++)
            for (int y = 0; y < image.cols; y++)
            {
              if (image.channels() == 3)
                blockImage[i].at<cv::Vec3b>(x - (int)image.rows / numofthread*i, y) = image.at<cv::Vec3b>(x, y);
              else
                blockImage[i].at<uchar>(x - (int)image.rows / numofthread*i, y) = image.at<uchar>(x, y);
            }
          imgnoise.medianFilterResult(blockImage[i], median_kernelsize, median_maxsize);
        }

        for (int i = 0; i < numofthread; i++)
        {
          for (int x = image.rows / numofthread*i; x < image.rows / numofthread*(i + 1); x++)
            for (int y = 0; y < image.cols; y++)
            {
              if (image.channels() == 3)
                image.at<cv::Vec3b>(x, y) = blockImage[i].at<cv::Vec3b>(x - (int)image.rows / numofthread*i, y);
              else
                image.at<uchar>(x, y) = blockImage[i].at<uchar>(x - (int)image.rows / numofthread*i, y);
            }
        }
        DWORD end_time = GetTickCount();
        if (flag == 1)
          running_time = (end_time - start_time) / 1000.0;
        else
          running_time += (end_time - start_time) / 1000.0;

        ui->widget_2->show();
        cv::imwrite(fileName.toStdString(), image);
        QPixmap *img_2 = new QPixmap(fileName);
        ui->label_2->resize(ui->widget_2->size());
        img_2->scaled(ui->label_2->size(), Qt::KeepAspectRatio);
        ui->label_2->setScaledContents(true);
        ui->label_2->setPixmap(*img_2);
        ui->textEdit->append(tr("OpenMP: Median filter takes %1s").arg(running_time));
      }

      if (ui->comboBox_2->currentIndex() == 5)
      {
        int type = ui->comboBox->currentIndex();
        if (theta == 0)
        {
          QMessageBox mesg;
          mesg.warning(this, "警告", "请输入参数！");
        }
        QString fileName_2 = "F:\\temp1.jpg";
        cv::Mat image_2 = cv::imread(fileName_2.toStdString(), 1);
        ui->widget_2->show();
        DWORD start_time = GetTickCount();
        imgnoise.rotate(image_2, theta, type, ui->spinBox->value());
        DWORD end_time = GetTickCount();
        if (flag == 1)
          running_time = (end_time - start_time) / 1000.0;
        else
          running_time += (end_time - start_time) / 1000.0;
        cv::imwrite(fileName_2.toStdString(), image_2);
        cv::imwrite(fileName.toStdString(), image_2);
        QPixmap *img_2 = new QPixmap(fileName);
        ui->label_2->resize(ui->widget_2->size());
        img_2->scaled(ui->label_2->size(), Qt::KeepAspectRatio);
        ui->label_2->setScaledContents(true);
        ui->label_2->setPixmap(*img_2);
        if (type == 1)
          ui->textEdit->append(tr("Single thread: Rotate takes %1s").arg(running_time));
        else if (type == 3)
          ui->textEdit->append(tr("OpenMP: Rotate takes %1s").arg(running_time));
      }

      if (ui->comboBox_2->currentIndex() == 6)
      {
        DFT dft_obj;

        int numofthread = ui->spinBox->value();
        QString fileName_2 = "F:\\temp1.jpg";
        cv::Mat image_2 = cv::imread(fileName_2.toStdString(), 0);
        ui->widget_2->show();
        DWORD start_time = GetTickCount();
        dft_obj.fft(image_2, 3, numofthread);
        DWORD end_time = GetTickCount();
        if (flag == 1)
          running_time = (end_time - start_time) / 1000.0;
        else
          running_time += (end_time - start_time) / 1000.0;
        cv::imwrite(fileName.toStdString(), image_2);
        QPixmap *img_2 = new QPixmap(fileName);
        ui->label_2->resize(ui->widget_2->size());
        img_2->scaled(ui->label_2->size(), Qt::KeepAspectRatio);
        ui->label_2->setScaledContents(true);
        ui->label_2->setPixmap(*img_2);
        ui->textEdit->append(tr("OpenMP: FFT takes %1s").arg(running_time));
      }
    }
    else if (ui->comboBox->currentIndex() == 4)
    {
      if (ui->comboBox_2->currentIndex() == 2)
      {
        if (filter_sigma == 0)
        {
          QMessageBox mesg;
          mesg.warning(this, "警告", "请输入参数！");
        }
        ui->widget_2->show();
        DWORD start_time = GetTickCount();
        imgnoise.gaussianFilter(image, filter_sigma, ui->comboBox->currentIndex());
        DWORD end_time = GetTickCount();
        running_time += (end_time - start_time) / 1000.0;
        cv::imwrite(fileName.toStdString(), image);
        QPixmap *img_2 = new QPixmap(fileName);
        ui->label_2->resize(ui->widget_2->size());
        img_2->scaled(ui->label_2->size(), Qt::KeepAspectRatio);
        ui->label_2->setScaledContents(true);
        ui->label_2->setPixmap(*img_2);
        ui->textEdit->append(tr("CUDA: Gaussian filter takes %1s").arg(running_time));
      }
      else
      {
        int type = ui->comboBox->currentIndex();
        if (theta == 0)
        {
          QMessageBox mesg;
          mesg.warning(this, "警告", "请输入参数！");
        }
        QString fileName_2 = "F:\\temp1.jpg";
        cv::Mat image_2 = cv::imread(fileName_2.toStdString(), 1);
        ui->widget_2->show();
        DWORD start_time = GetTickCount();
        imgnoise.rotate(image_2, theta, type, ui->spinBox->value());
        DWORD end_time = GetTickCount();
        if (flag == 1)
          running_time = (end_time - start_time) / 1000.0;
        else
          running_time += (end_time - start_time) / 1000.0;
        cv::imwrite(fileName_2.toStdString(), image_2);
        cv::imwrite(fileName.toStdString(), image_2);
        QPixmap *img_2 = new QPixmap(fileName);
        ui->label_2->resize(ui->widget_2->size());
        img_2->scaled(ui->label_2->size(), Qt::KeepAspectRatio);
        ui->label_2->setScaledContents(true);
        ui->label_2->setPixmap(*img_2);
        ui->textEdit->append(tr("CUDA: Rotate takes %1s").arg(running_time));
      }     
    }
    else if (ui->comboBox->currentIndex() == 5)
    {
      if (ui->comboBox_2->currentIndex() == 2)
      {
        if (filter_sigma == 0)
        {
          QMessageBox mesg;
          mesg.warning(this, "警告", "请输入参数！");
        }
        ui->widget_2->show();
        DWORD start_time = GetTickCount();
        imgnoise.gaussianFilter(image, filter_sigma, ui->comboBox->currentIndex());
        DWORD end_time = GetTickCount();
        running_time += (end_time - start_time) / 1000.0;
        cv::imwrite(fileName.toStdString(), image);
        QPixmap *img_2 = new QPixmap(fileName);
        ui->label_2->resize(ui->widget_2->size());
        img_2->scaled(ui->label_2->size(), Qt::KeepAspectRatio);
        ui->label_2->setScaledContents(true);
        ui->label_2->setPixmap(*img_2);
        ui->textEdit->append(tr("OpenCL: Gaussian filter takes %1s").arg(running_time));
      }
      else
      {
        int type = ui->comboBox->currentIndex();
        if (theta == 0)
        {
          QMessageBox mesg;
          mesg.warning(this, "警告", "请输入参数！");
        }
        QString fileName_2 = "F:\\temp1.jpg";
        cv::Mat image_2 = cv::imread(fileName_2.toStdString(), 1);
        ui->widget_2->show();
        DWORD start_time = GetTickCount();
        imgnoise.rotate(image_2, theta, type, ui->spinBox->value());
        DWORD end_time = GetTickCount();
        if (flag == 1)
          running_time = (end_time - start_time) / 1000.0;
        else
          running_time += (end_time - start_time) / 1000.0;
        cv::imwrite(fileName_2.toStdString(), image_2);
        cv::imwrite(fileName.toStdString(), image_2);
        QPixmap *img_2 = new QPixmap(fileName);
        ui->label_2->resize(ui->widget_2->size());
        img_2->scaled(ui->label_2->size(), Qt::KeepAspectRatio);
        ui->label_2->setScaledContents(true);
        ui->label_2->setPixmap(*img_2);
        ui->textEdit->append(tr("OpenCL: Rotate takes %1s").arg(running_time));
      }
    }
  }

}


void MainWindow::on_pushButton_3_clicked()
{
  ImageNoise imgnoise;
  QString fileName = "F:\\temp1.jpg";
  QString fileName_2 = "F:\\temp2.jpg";

  cv::Mat image = cv::imread(fileName.toStdString(), 1);

  double means = -1, sigma = -1, SNR = -1;
  int k = 0;
  k = ui->noiseCoefficient->text().toInt();
  means = ui->gaussian_means->text().toDouble();
  sigma = ui->gaussian_sigma->text().toDouble();
  SNR = ui->line_SNR->text().toDouble();

  if (ui->comboBox->currentIndex() == 1)
  {
    if (ui->comboBox_3->currentIndex() == 1)
    {
      if (k == 0 || means == -1 || sigma == -1)
      {
        QMessageBox mesg;
        mesg.warning(this, "警告", "请输入参数！");
      }
      ui->widget_2->show();
      DWORD start_time = GetTickCount();
      imgnoise.addGaussianNoise(image, k, means, sigma, ui->comboBox->currentIndex());
      DWORD end_time = GetTickCount();
      double running_time = (end_time - start_time) / 1000.0;
      cv::imwrite(fileName_2.toStdString(), image);
      QPixmap *img_2 = new QPixmap(fileName_2);
      ui->label_2->resize(ui->widget_2->size());
      img_2->scaled(ui->label_2->size(), Qt::KeepAspectRatio);
      ui->label_2->setScaledContents(true);
      ui->label_2->setPixmap(*img_2);
      ui->textEdit->append(tr("Single thread: Add Gaussian noise take %1s").arg(running_time));
    }
    else if (ui->comboBox_3->currentIndex() == 2)
    {
      if (SNR == -1)
      {
        QMessageBox mesg;
        mesg.warning(this, "警告", "请输入参数！");
      }
      ui->widget_2->show();
      DWORD start_time = GetTickCount();
      imgnoise.addSaultandPepperNoise(image, SNR);
      DWORD end_time = GetTickCount();
      double running_time = (end_time - start_time) / 1000.0;
      cv::imwrite(fileName_2.toStdString(), image);
      QPixmap *img_2 = new QPixmap(fileName_2);
      ui->label_2->resize(ui->widget_2->size());
      img_2->scaled(ui->label_2->size(), Qt::KeepAspectRatio);
      ui->label_2->setScaledContents(true);
      ui->label_2->setPixmap(*img_2);
      ui->textEdit->append(tr("Single thread: Add SaultandPepper noise take %1s").arg(running_time));
    }
  }
  else if (ui->comboBox->currentIndex() == 2)
  {

    DWORD start_time = GetTickCount();
    int numofthread = ui->spinBox->value();
    MyThread *mythread = new MyThread[numofthread]();
    cv::Mat *blockImage = new cv::Mat[numofthread];

    for (int i = 0; i < numofthread; i++)
    {
      blockImage[i] = cv::Mat::zeros(image.rows / numofthread, image.cols, image.type());
      for (int x = image.rows / numofthread*i; x < image.rows / numofthread*(i + 1); x++)
        for (int y = 0; y < image.cols; y++)
        {
          if (image.channels() == 3)
            blockImage[i].at<cv::Vec3b>(x - (int)image.rows / numofthread*i, y) = image.at<cv::Vec3b>(x, y);
          else
            blockImage[i].at<uchar>(x - (int)image.rows / numofthread*i, y) = image.at<uchar>(x, y);
        }
    }

    for (int i = 0; i < numofthread; i++)
    {
      mythread[i].start();
      mythread[i].startThread_noise(blockImage[i], ui->comboBox_3->currentIndex(), means, sigma, SNR, k);
    }

    for (int i = 0; i < numofthread; i++)
    {
      for (int x = image.rows / numofthread*i; x < image.rows / numofthread*(i + 1); x++)
        for (int y = 0; y < image.cols; y++)
        {
          if (image.channels() == 3)
            image.at<cv::Vec3b>(x, y) = blockImage[i].at<cv::Vec3b>(x - (int)image.rows / numofthread*i, y);
          else
            image.at<uchar>(x, y) = blockImage[i].at<uchar>(x - (int)image.rows / numofthread*i, y);
        }
    }
    DWORD end_time = GetTickCount();
    double running_time = (end_time - start_time) / 1000.0;
    ui->widget_2->show();
    cv::imwrite(fileName_2.toStdString(), image);
    QPixmap *img_2 = new QPixmap(fileName_2);
    ui->label_2->resize(ui->widget_2->size());
    img_2->scaled(ui->label_2->size(), Qt::KeepAspectRatio);
    ui->label_2->setScaledContents(true);
    ui->label_2->setPixmap(*img_2);
    if (ui->comboBox_3->currentIndex() == 1)
      ui->textEdit->append(tr("Multi thread: Add Gaussian noise takes %1s").arg(running_time));
    else
      ui->textEdit->append(tr("Multi thread: Add SaultanPepper noise takes %1s").arg(running_time));
  }
  else if (ui->comboBox->currentIndex() == 3)
  {
    int numofthread = ui->spinBox->value();
    cv::Mat *blockImage = new cv::Mat[numofthread];
    if (ui->comboBox_3->currentIndex() == 1)
    {
      if (k == 0 || means == -1 || sigma == -1)
      {
        QMessageBox mesg;
        mesg.warning(this, "警告", "请输入参数！");
      }
      DWORD start_time = GetTickCount();
      omp_set_num_threads(numofthread);
#pragma omp parallel for
      for (int i = 0; i < numofthread; i++)
      {
        blockImage[i] = cv::Mat::zeros(image.rows / numofthread, image.cols, image.type());
        for (int x = image.rows / numofthread*i; x < image.rows / numofthread*(i + 1); x++)
          for (int y = 0; y < image.cols; y++)
          {
            if (image.channels() == 3)
              blockImage[i].at<cv::Vec3b>(x - (int)image.rows / numofthread*i, y) = image.at<cv::Vec3b>(x, y);
            else
              blockImage[i].at<uchar>(x - (int)image.rows / numofthread*i, y) = image.at<uchar>(x, y);
          }
        imgnoise.addGaussianNoise(blockImage[i], k, means, sigma, ui->comboBox->currentIndex());
      }

      omp_set_num_threads(numofthread);
#pragma omp parallel for
      for (int i = 0; i < numofthread; i++)
      {
        for (int x = image.rows / numofthread*i; x < image.rows / numofthread*(i + 1); x++)
          for (int y = 0; y < image.cols; y++)
          {
            if (image.channels() == 3)
              image.at<cv::Vec3b>(x, y) = blockImage[i].at<cv::Vec3b>(x - (int)image.rows / numofthread*i, y);
            else
              image.at<uchar>(x, y) = blockImage[i].at<uchar>(x - (int)image.rows / numofthread*i, y);
          }
      }
      DWORD end_time = GetTickCount();
      double running_time = (end_time - start_time) / 1000.0;
      ui->widget_2->show();
      cv::imwrite(fileName_2.toStdString(), image);
      QPixmap *img_2 = new QPixmap(fileName_2);
      ui->label_2->resize(ui->widget_2->size());
      img_2->scaled(ui->label_2->size(), Qt::KeepAspectRatio);
      ui->label_2->setScaledContents(true);
      ui->label_2->setPixmap(*img_2);
      ui->textEdit->append(tr("OpenMP: Add Gaussian noise takes %1s").arg(running_time));
    }

    if (ui->comboBox_3->currentIndex() == 2)
    {
      if (SNR == -1)
      {
        QMessageBox mesg;
        mesg.warning(this, "警告", "请输入参数！");
      }
      DWORD start_time = GetTickCount();
      omp_set_num_threads(numofthread);
#pragma omp parallel for
      for (int i = 0; i < numofthread; i++)
      {
        blockImage[i] = cv::Mat::zeros(image.rows / numofthread, image.cols, image.type());
        for (int x = image.rows / numofthread*i; x < image.rows / numofthread*(i + 1); x++)
          for (int y = 0; y < image.cols; y++)
          {
            if (image.channels() == 3)
              blockImage[i].at<cv::Vec3b>(x - (int)image.rows / numofthread*i, y) = image.at<cv::Vec3b>(x, y);
            else
              blockImage[i].at<uchar>(x - (int)image.rows / numofthread*i, y) = image.at<uchar>(x, y);
          }
        imgnoise.addSaultandPepperNoise(blockImage[i], SNR);
      }

      for (int i = 0; i < numofthread; i++)
      {
        for (int x = image.rows / numofthread*i; x < image.rows / numofthread*(i + 1); x++)
          for (int y = 0; y < image.cols; y++)
          {
            if (image.channels() == 3)
              image.at<cv::Vec3b>(x, y) = blockImage[i].at<cv::Vec3b>(x - (int)image.rows / numofthread*i, y);
            else
              image.at<uchar>(x, y) = blockImage[i].at<uchar>(x - (int)image.rows / numofthread*i, y);
          }
      }
      DWORD end_time = GetTickCount();
      double running_time = (end_time - start_time) / 1000.0;
      ui->widget_2->show();
      cv::imwrite(fileName_2.toStdString(), image);
      QPixmap *img_2 = new QPixmap(fileName_2);
      ui->label_2->resize(ui->widget_2->size());
      img_2->scaled(ui->label_2->size(), Qt::KeepAspectRatio);
      ui->label_2->setScaledContents(true);
      ui->label_2->setPixmap(*img_2);
      ui->textEdit->append(tr("OpenMP: Add SaultandPepper noise takes %1s").arg(running_time));
    }
  }
}

void MainWindow::on_pushButton_4_clicked()
{
  ImageNoise imgnoise;
  QString fileName = "F:\\temp1.jpg";
  cv::Mat image = cv::imread(fileName.toStdString(), 1);
  cv::imshow("src", image);
  ui->doubleSpinBox->setSingleStep(0.1);
  double scale = ui->doubleSpinBox->value();
  int type = ui->comboBox->currentIndex();
  int numofthread = ui->spinBox->value();
  if (type == 1 || type == 3 || type == 4 || type == 5)
  {
    DWORD start_time = GetTickCount();
    imgnoise.BicubicInterpolation(image, scale, scale, type, numofthread);
    DWORD end_time = GetTickCount();
    double running_time = (end_time - start_time) / 1000.0;
    if (type == 1)
      ui->textEdit->append(tr("Single thread: Scale takes %1s").arg(running_time));
    else if (type == 3)
      ui->textEdit->append(tr("OpenMP: Scale takes %1s").arg(running_time));
    else if (type == 4)
      ui->textEdit->append(tr("CUDA: Scale takes %1s").arg(running_time));
    else if (type == 5)
      ui->textEdit->append(tr("OpenCL: Scale takes %1s").arg(running_time));
    cv::imwrite(fileName.toStdString(), image);
  }
  else if (type == 2)
  {
    MyThread *mythread = new MyThread[numofthread]();
    cv::Mat dst;
    DWORD start_time = GetTickCount();
    dst.create(image.rows*scale, image.cols*scale, image.type());

    for (int i = 0; i < numofthread; i++)
    {
      mythread[i].start();
      mythread[i].startThread_scale(image, dst, numofthread, i, scale, scale);
    }
    DWORD end_time = GetTickCount();
    double running_time = (end_time - start_time) / 1000.0;
    cv::imshow("output", dst);
    ui->textEdit->append(tr("Multi thread: Scale takes %1s").arg(running_time));
    cv::imwrite(fileName.toStdString(), dst);
  }
}


void MainWindow::on_comboBox_currentIndexChanged(int index)
{

}


void MainWindow::on_comboBox_2_currentIndexChanged(int index)
{
  if (ui->comboBox_2->currentIndex() == 2)
    ui->tabWidget->setCurrentWidget(ui->tab);
  else if (ui->comboBox_2->currentIndex() == 3)
    ui->tabWidget->setCurrentWidget(ui->tab_2);
  else if (ui->comboBox_2->currentIndex() == 4)
    ui->tabWidget->setCurrentWidget(ui->tab_7);
  else if (ui->comboBox_2->currentIndex() == 5)
    ui->tabWidget->setCurrentWidget(ui->tab_5);
}


void MainWindow::on_comboBox_3_currentIndexChanged(int index)
{

  if (ui->comboBox_3->currentIndex() == 1)
    ui->tabWidget->setCurrentWidget(ui->tab_3);
  else if (ui->comboBox_3->currentIndex() == 2)
    ui->tabWidget->setCurrentWidget(ui->tab_4);
}



void MainWindow::on_linearSlider_valueChanged(int value)
{

}


