/*
 * NaiveGaussianHOGPC.cpp
 *
 *  Created on: Apr 2, 2017
 *      Author: timo
 */

#include <opencv/ml.h>
#include <opencv2/objdetect.hpp>
#include "NaiveGaussianHOGPC.h"

NaiveGaussianHOG_PC::NaiveGaussianHOG_PC(vector<float> means, vector<float> variances, vector<int> mask) {
	this->means=means;
	this->variances=variances;
	this->mask=mask;
}

NaiveGaussianHOG_PC::~NaiveGaussianHOG_PC() {
	// TODO Auto-generated destructor stub
}

string NaiveGaussianHOG_PC::classify(Mat image) {
	if(image.cols != 40 || image.rows != 40) {
		HOGDescriptor hog;
		hog.winSize = Size(40,40);
		vector<float> descriptors;
		vector< Point > locations;
		hog.compute(image, descriptors, Size(8,8), Size(0,0), locations);
	} else {
		cout << "image has size differenct of 40x40!" << endl;
	}
	return "TODO";
}

bool NaiveGaussianHOG_PC::isObject(Mat hog_values) {
	bool res = true;
	if(hog_values.rows == mask.size()) {
		int i=0;
		for(i=0;i<mask.size() && res;++i) {
			if(mask[i]==1) {
				float val = hog_values.at<float>(i,0);
				res = res && (means[i] - 3*variances[i] <= val) && (val <= means[i] + 3*variances[i]);
			}
		}
		cout << i << "  " << hog_values.at<float>(i,0) << endl;
	} else {
		res = false;
		cout << "ERROR " << mask.size() << " != " << hog_values.size() << endl;
	}
	return res;
}

