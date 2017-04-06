/*
 * Classifier.cpp
 *
 *  Created on: Feb 28, 2017
 *      Author: timo
 *      Version: 0.02
 */

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "ParkingClassifier.h"
#include "util.hpp"

ParkingClassifier::ParkingClassifier()
: parking_spots(), output_path(""), show_progress(false){
}

ParkingClassifier::~ParkingClassifier() {
}

void ParkingClassifier::set_parking_spots(vector<Parkplatz*> parking_spots) {
	this->parking_spots = parking_spots;
}

void ParkingClassifier::set_output_path(string output_path) {
	this->output_path = output_path;
}

vector<int> ParkingClassifier::test(ParkingDatasetDescription parkingDatasetDescription) {
	vector<int> res;
	int no_false_classified_false = 0;
	int no_false_classified_true = 0;
	int no_true_classified_false = 0;
	int no_true_classified_true = 0;
	vector<string> filenames = get_keys(parkingDatasetDescription.get_occupation());
	const int no_samples = filenames.size();

	for(int i=0;i<no_samples;++i) {
		if(show_progress)
			cout << "Testing image " << filenames[i] << " [" << i << "/" << no_samples << "]" << endl;
		Mat img = imread(parkingDatasetDescription.get_directory() + "/" + filenames[i], CV_LOAD_IMAGE_COLOR);
		Mat imgGray = img;
//		cvtColor(img, imgGray, COLOR_BGR2GRAY );
		Mat overlay = Mat(img.size(), CV_8UC3, Scalar(0,0,0));
		string responseRow = parkingDatasetDescription.get_occupation()[filenames[i]];
		string classifications = classify_test(imgGray, responseRow);
		for(int j = 0; j < responseRow.length(); ++j) {
			char response = responseRow.at(j);
			if(classifications[j]=='e') {
				if(response=='e') {
					++no_true_classified_true;
				} else {
					++no_false_classified_true;
				}
				if(this->output_path != "") {
					Point corners[4];
					parkingDatasetDescription.get_parking_spots()[j]->getCorners(corners);
					fillConvexPoly(overlay, corners, 4, Scalar(0,255,0));
				}
			} else {// classification is 'o'.
				if(response=='e') {
					++no_true_classified_false;
				} else {
					++no_false_classified_false;
				}
				if(this->output_path != "") {
					Point corners[4];
					parkingDatasetDescription.get_parking_spots()[j]->getCorners(corners);
					fillConvexPoly(overlay, corners, 4, Scalar(0,0,255));
				}
			}
		}
		if(output_path != "") {
			addWeighted(img,0.5,overlay,0.5,0,img);
			imwrite(output_path + "/" + filenames[i], img);
		}
	}

	res.push_back(no_false_classified_false);
	res.push_back(no_false_classified_true);
	res.push_back(no_true_classified_false);
	res.push_back(no_true_classified_true);
	return res;
}

string ParkingClassifier::classify_test(Mat image, string targets) {
	return classify(image);
}

void ParkingClassifier::set_show_progress(bool show_progress) {
	this->show_progress = show_progress;
}
