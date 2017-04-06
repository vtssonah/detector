/*
 * SVMParkingClassifier.cpp
 *
 *  Created on: Feb 28, 2017
 *      Author: timo
 */

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/objdetect.hpp>

#include "SVMParkingClassifier.h"
#include "util.hpp"

SVM_ParkingClassifier::SVM_ParkingClassifier(Size window_size, bool show_detections) {
	this->winSize = window_size;
	this->winStride = Size(8,8);
	this->show_detections = show_detections;
	this->out_index = 0;
}

SVM_ParkingClassifier::~SVM_ParkingClassifier() {
	// TODO Auto-generated destructor stub
}

string SVM_ParkingClassifier::classify(Mat image) {
	string res;
	HOGDescriptor hog;
	hog.winSize = winSize;
	vector< float > hog_detector;
	get_svm_detector(svm, hog_detector);
	hog.setSVMDetector(hog_detector);
	Mat imgGray;
	cvtColor(image, imgGray, COLOR_BGR2GRAY);
	equalizeHist(imgGray, imgGray);
	detected_locations.clear();
	vector<Rect> locations;
	hog.detectMultiScale(imgGray, locations, 0, this->winStride);

	for(int p=0;p<this->parking_spots.size();++p){
		Mat boundary;
		Parkplatz* parkplatz = this->parking_spots[p];
		parkplatz->getBoundary(&boundary);
		Mat warpedParkingSpot;
		string vote = "e";
		vector<Rect> detections_on_spot;
		for(int i = 0; i < locations.size(); ++i) {
			Rect r = locations[i];

			// Center point of the vehicle
			Point center(r.x + r.width / 2, r.y + r.height / 2);
			if(pointPolygonTest(boundary, center, false) >= 0) {
				vote = "o";
				detections_on_spot.push_back(r);
				break;
			}
		}
		res.append(vote);
		detected_locations.push_back(detections_on_spot);
	}
	if(show_detections) {
		Mat img = image.clone();
		draw_locations(img, locations, Scalar(0, 255, 0));
		this->detection_images.push_back(img);
		imwrite("detections.png", img);
	}
	return res;
}

string SVM_ParkingClassifier::classify_test(Mat image, string targets) {
	string response = classify(image);
	for(int i=0;i<targets.size();++i){
		if(targets[i] != response[i]) {
			for(int j=0;j<detected_locations[i].size();++j) {
				stringstream filename;
				filename << "hard_negatives/hard_negative_";
				filename << out_index;
				filename << "_";
				filename << i;
				filename << "_";
				filename << j;
				filename << ".png";
				cout << filename.str() << endl;
				imwrite(filename.str(), image(detected_locations[i][j]));
			}
		}
	}
	++out_index;
	return response;
}


void SVM_ParkingClassifier::set_SVM(Ptr<SVM> svm) {
	this->svm = svm;
	this->detection_images.clear();
}

void SVM_ParkingClassifier::set_window_size(Size winSize) {
	this->winSize = winSize;
	this->detection_images.clear();
}
void SVM_ParkingClassifier::set_window_stride(Size winStride) {
	this->winStride = winStride;
}

vector<Mat> SVM_ParkingClassifier::get_detection_images() {
	return this->detection_images;
}

vector< vector< Rect > > SVM_ParkingClassifier::get_detected_locations() {
	return this->detected_locations;
}

