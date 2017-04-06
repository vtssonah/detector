/*
 * Classifier.h
 *
 *  Created on: Feb 28, 2017
 *      Author: timo
 *      Version: 0.02
 */

#ifndef PARKINGCLASSIFIER_H_
#define PARKINGCLASSIFIER_H_

#include <vector>
#include <opencv/cv.h>
#include "Parkplatz.hpp"
#include "ParkingDatasetDescription.h"

using namespace std;
using namespace cv;

class ParkingClassifier {
protected:
	vector<Parkplatz*> parking_spots;//TODO: Should be vector<Parkplatz>.
	string output_path;
	bool show_progress;
public:
	ParkingClassifier();
	virtual ~ParkingClassifier();
	void set_parking_spots(vector<Parkplatz*> parking_spots);
	void set_output_path(string output_path);
	virtual string classify(Mat image)=0;
	virtual string classify_test(Mat image, string targets);
	vector<int> test(ParkingDatasetDescription parkingDatasetDescription);
	void set_show_progress(bool show_progress);
};

#endif /* PARKINGCLASSIFIER_H_ */
