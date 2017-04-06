/*
 * HarrisPC.cpp
 *
 *  Created on: Mar 20, 2017
 *      Author: timo
 */

#include "HarrisPC.h"
#include <opencv2/imgproc.hpp>


HarrisPC::HarrisPC(Size window_size) : threshold(0){
	this->winSize = window_size;
}

HarrisPC::~HarrisPC() {
	// TODO Auto-generated destructor stub
}

string HarrisPC::classify(Mat image) {
	string res;
	Mat src_gray;
	Mat dst, dst_norm, dst_norm_scaled;
	dst = Mat::zeros( image.size(), CV_32FC1 );
	cvtColor( image, src_gray, COLOR_BGR2GRAY );
	equalizeHist( src_gray, src_gray );
	int blockSize = 4;
	int apertureSize = 5;
	double k = 0.04;

	/// Detecting corners
	cornerHarris( src_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT );
	normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
	convertScaleAbs( dst_norm, dst_norm_scaled );
	/// Drawing a circle around corners
	for(int p=0;p<this->parking_spots.size();++p){
		Mat boundary;
		Parkplatz* parkplatz = this->parking_spots[p];
		parkplatz->getBoundary(&boundary);
		string vote = "e";
		for( int j = 0; j < dst_norm.rows && vote == string("e"); j++ ){
			for( int i = 0; i < dst_norm.cols && vote == string("e"); i++ ) {
				if( (int) dst_norm.at<float>(j,i) > threshold  && pointPolygonTest(boundary, Point(i,j), false) >= 0) {
					vote = "o";
				}
			}
		}
		res.append(vote);
	}
	return res;
}

void HarrisPC::set_threshold(int threshold) {
	this->threshold = threshold;
}
