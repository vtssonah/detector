/*
 * HarrisPC.h
 *
 *  Created on: Mar 20, 2017
 *      Author: timo
 */

#ifndef HARRISPC_H_
#define HARRISPC_H_

#include "ParkingClassifier.h"

class HarrisPC : public ParkingClassifier {
private:
	Size winSize;
	int threshold;
public:
	HarrisPC(Size window_size);
	virtual ~HarrisPC();
	string classify(Mat image);
	void set_threshold(int threshold);
};

#endif /* HARRISPC_H_ */
