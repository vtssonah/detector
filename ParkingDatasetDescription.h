/*
 * DatasetDescription.h
 *
 *  Created on: Feb 26, 2017
 *      Author: timo
 * @version 0.01
 */

#ifndef PARKINGDATASETDESCRIPTION_H_
#define PARKINGDATASETDESCRIPTION_H_

#include <opencv2/ml.hpp>
#include <map>
#include <string>
#include "Parkplatz.hpp"

using namespace std;
using namespace cv;
using namespace cv::ml;


class ParkingDatasetDescription {
private:
	ParkingDatasetDescription(string directory, vector<Parkplatz*> parkingSpots, map<string,string> occupation);
	/**
	 * Directory of all training images. It is assumed that all files are contained in the same directory.
	 */
	string directory;
	/**
	 * The locations of parking spots in the images.
	 */
	vector<Parkplatz*> parkingSpots;
	/**
	 * Maps file name to occupation.
	 * Example (filename1.jpg contains 9 parking spots):
	 *    filename1.jpg oeoeooooe
	 */
	map<string, string> occupation;
	int count_occupation(char c);
public:
	virtual ~ParkingDatasetDescription();
	/**
	 * Loads a dataset description form a file with usual ending as .lbl.
	 */
	static ParkingDatasetDescription load(string file);
	void save(string file);
	string get_directory();
	string get_absolute_path_to_file(string file);
	vector<Parkplatz*> get_parking_spots();
	map<string, string> get_occupation();
	Ptr<TrainData> to_TrainData();
	void load_images_and_labels(vector<Mat>& images, vector<string>& labels);
	Ptr<TrainData> get_HOG_features(Size image_size);
	int get_no_empty_spots();
	int get_no_occupied_spots();
};

#endif /* PARKINGDATASETDESCRIPTION_H_ */
