#ifndef DFT_H
#define DFT_H

#include<opencv2/core.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>
#include<iostream>

#define intsize sizeof(int)
#define complexsize sizeof(complex)
#define PI 3.1415926535

using namespace cv;
using namespace std;

struct myComplex{
    float real;
    float img;
};

class DFT{
public:
    DFT();
    myComplex Add(myComplex, myComplex);
    myComplex Mul(myComplex, myComplex);
    void resizeImage(Mat &);
    int if_binaryNum(int);
    void reverse(int *, int);
    void fft(Mat &, int, int);
};




#endif // DFT_H
