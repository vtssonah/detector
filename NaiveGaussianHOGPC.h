/*
 * NaiveGaussianHOGPC.h
 *
 *  Created on: Apr 2, 2017
 *      Author: timo
 */

#ifndef NAIVEGAUSSIANHOGPC_H_
#define NAIVEGAUSSIANHOGPC_H_
#include <vector>
#include "ParkingClassifier.h"

class NaiveGaussianHOG_PC: public ParkingClassifier {
protected:
	vector<float> means;
	vector<float> variances;
	vector<int> mask;
public:
	NaiveGaussianHOG_PC(vector<float> means, vector<float> variances, vector<int> mask);
	virtual ~NaiveGaussianHOG_PC();
	string classify(Mat image);
	bool isObject(Mat hog_values);
};

#endif /* NAIVEGAUSSIANHOGPC_H_ */
