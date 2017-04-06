
#ifndef IMAGE_ALG_H_
#define IMAGE_ALG_H_

#include "opencv/cv.h"
#include "Parkplatz.hpp"

using namespace std;
using namespace cv;

double calreldev(Mat frame);
double calreldev(Mat frame, Mat mask);
int getcarlength(Parkplatz* spot, bool& leftright);
bool my_comp(const Point& a, const Point& b);
vector<Point> sortPoints(vector<Point> unsorted);
vector<Point*> split_pf(vector<Point> pfield,int factor, bool leftright, float& frac_mid,bool warped);
vector<Point> turnPoints(vector<Point> normal);
vector <int> get_histo(Mat frame,float& mean, float& stdev);
Mat extr_Spec(Mat frame,int low,int high);
Mat four_point_transform(Mat frame,vector<Point>pts,vector<Point>& pfcorners_normal_warped,Mat& M);
vector<double> calcmopsstrips(Mat overlay,Mat frame, vector<Point*> mops_strips,double reldev_pf,int mopstreshhold);
int calciffree_1b1(vector<double> mops_strips_result,float striplength,int mopstreshhold,int carlength);
int calcfreespots(vector<double> mops_strips_result,float striplength,int mopstreshhold,int carlength);

#endif
