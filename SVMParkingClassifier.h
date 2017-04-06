/*
 * SVMParkingClassifier.h
 *
 *  Created on: Feb 28, 2017
 *      Author: timo
 */

#ifndef SVMPARKINGCLASSIFIER_H_
#define SVMPARKINGCLASSIFIER_H_

#include "ParkingClassifier.h"

class SVM_ParkingClassifier : public ParkingClassifier {
private:
	Ptr<SVM> svm;
	Size winSize;
	Size winStride;
	bool show_detections;
	vector<Mat> detection_images;
	vector< vector< Rect > > detected_locations;
	int out_index;
public:
	SVM_ParkingClassifier(Size window_size, bool show_detections);
	virtual ~SVM_ParkingClassifier();
	string classify(Mat image);
	string classify_test(Mat image, string targets);
	void set_SVM(Ptr<SVM> svm);
	void set_window_size(Size winSize);
	void set_window_stride(Size winStride);
	vector<Mat> get_detection_images();
	vector< vector< Rect > > get_detected_locations();
};

#endif /* SVMPARKINGCLASSIFIER_H_ */
