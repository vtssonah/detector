

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include "ParkingDatasetDescription.h"
#include "util.hpp"

const string PARKING_SPOTS = "PARKING_SPOTS";
const string IMAGE_MAP = "IMAGE_MAP";
const string DIRECTORY = "DIRECTORY";
const string LAST_IMAGE = "LAST_IMAGE";



ParkingDatasetDescription::ParkingDatasetDescription(string directory, vector<Parkplatz*> parkingSpots, map<string,string> occupation) {
	this->directory=directory;
	this->parkingSpots=parkingSpots;
	this->occupation=occupation;
}

ParkingDatasetDescription::~ParkingDatasetDescription() {
	// TODO Auto-generated destructor stub
}

ParkingDatasetDescription ParkingDatasetDescription::load(string file) {
	vector<Parkplatz*> parking_spots;
	string directory;
	map<string, string> occupationMap;
	string content = read_file(file.c_str());
	vector<string> lines = ssplit(content, '\n');
	int parseState = 0;
	vector<string> occupation;
	vector<string> candidate;
	for(int i=0;i<lines.size();++i){
		if(lines[i].length() > 0 && lines[i].at(0) != '#') {
//			cout << "parseState is " << parseState << endl;
			switch(parseState) {
			case 0:
				if(lines[i] == PARKING_SPOTS) {// one of the following lines contains the parking spot definitions.
					parseState = 1;
				} else if(lines[i] == IMAGE_MAP) {// The following lines have a DIRECTORY and a LAST_IMAGE followed by occupation.
					parseState = 2;
				}
				break;
			case 1:
				parking_spots = Parkplatz::getParkingsSpotsFromString(lines[i]);
				parseState = 0;
				break;
			case 2:
				candidate = ssplit(lines[i],'=');
				if(candidate.size()==2) {
					if(candidate[0]==DIRECTORY)
						directory = candidate[1];
					else if(candidate[0]==LAST_IMAGE)
						parseState = 3;
				}
				break;
			case 3:
				occupation = ssplit(lines[i],' ');
				if(occupation.size()==3 && occupation[1]=="0") {// occupation[1] is the pointer to the parking spot collection.
					occupationMap[occupation[0]] = occupation[2];
				}
				break;
			default:
				break;
			}
		}
	}
	return ParkingDatasetDescription(directory, parking_spots, occupationMap);
}

void ParkingDatasetDescription::save(string file) {
	FILE* f = fopen(file.c_str(),  "wb");
	fwrite((PARKING_SPOTS+"\n").c_str(), 1, PARKING_SPOTS.length()+1, f);
	string parking_spot_str;
	for(int i=0;i<parkingSpots.size()-1;++i) {
		parking_spot_str += parkingSpots[i]->getString2() + " ";
	}
	parking_spot_str += parkingSpots[parkingSpots.size()-1]->getString2() + "\n\n" + IMAGE_MAP + "\n";
	fwrite(parking_spot_str.c_str(), 1, parking_spot_str.size(), f);
	string directory_str(DIRECTORY + "=" + directory + "\n");
	fwrite(directory_str.c_str(), 1, directory_str.length(), f);
	string last_image_str(LAST_IMAGE + "=" + (*occupation.begin()).first + "\n");
	fwrite(last_image_str.c_str(), 1, last_image_str.length(), f);
	for(map<string,string>::iterator it = occupation.begin(); it != occupation.end(); ++it) {
		string occupation_line(it->first + " 0 " + it->second + "\n");
		fwrite(occupation_line.c_str(), 1, occupation_line.length(), f);
	}
	fclose(f);
}

string ParkingDatasetDescription::get_directory() {
	return this->directory;
}

string ParkingDatasetDescription::get_absolute_path_to_file(string file) {
	return directory + "/" + file;
}

vector<Parkplatz*> ParkingDatasetDescription::get_parking_spots() {
	return this->parkingSpots;
}

map<string,string> ParkingDatasetDescription::get_occupation() {
	return this->occupation;
}

Ptr<TrainData> ParkingDatasetDescription::to_TrainData() {
	vector<Mat> images;
	vector<string> labels;
	load_images_and_labels(images, labels);
	// Convert labels in {"o","e"} to labels in {1,-1}.
	vector< vector<int> > labelsConverted;
	for(int i=0;i<labels.size();++i){
		vector<int> labelsRow;
		for(int j=0;j<labels[i].length();++j){
//			labelsRow.push_back(labels[i].at(j)=='o'?LABEL_POSITIVE:LABEL_NEGATIVE);
		}
		labelsConverted.push_back(labelsRow);
	}
cout << images.size() << endl;
cout << labelsConverted.size() << endl;
	return TrainData::create(images, ROW_SAMPLE, labelsConverted);
}

void ParkingDatasetDescription::load_images_and_labels(vector<Mat>& images, vector<string>& labels) {
	for(map<string,string>::iterator it = occupation.begin(); it != occupation.end(); ++it) {
		images.push_back(imread(get_directory() + "/" + it->first, CV_LOAD_IMAGE_COLOR));
		labels.push_back(it->second);
	}
}

Ptr<TrainData> ParkingDatasetDescription::get_HOG_features(Size image_size) {
	vector<Mat> images;
	vector< string > labels;
	load_images_and_labels(images, labels);
	vector< Mat > pos_lst;
	vector< Mat > neg_lst;
	vector<Rect> boundaries;

	for(int j = 0; j < this->parkingSpots.size(); ++j) {
		Mat boundary;
//		cout << j << endl;
		parkingSpots[j]->getBoundary(&boundary);
//		cout << boundary << endl;
		boundaries.push_back(boundingRect(boundary));
	}
	// crop spot snippets and store in corresponding list.
	for(int i = 0; i < images.size(); ++i) {
		for(int j = 0; j < labels[i].length(); ++j) {
			Mat snippet(images[i], boundaries[j]);
			resize(snippet, snippet, image_size);
			if(labels[i].at(j)=='o')
				pos_lst.push_back(snippet.clone());
			else
				neg_lst.push_back(snippet.clone());
		}
	}
	vector< Mat > gradient_lst;
	compute_hog(pos_lst, gradient_lst, image_size);
	compute_hog(neg_lst, gradient_lst, image_size);
	vector<int> labelsSVM;
	labelsSVM.assign(pos_lst.size(), +1);
	labelsSVM.insert(labelsSVM.end(), neg_lst.size(), -1);
	cout << gradient_lst.size() << endl;
	cout << labelsSVM.size() << endl;
	Mat train_data;
	convert_to_ml(gradient_lst, train_data);
	return TrainData::create(train_data, ROW_SAMPLE, labelsSVM);
}

int ParkingDatasetDescription::count_occupation(char c) {
	int cnt = 0;
	for(map<string,string>::iterator it = occupation.begin(); it != occupation.end(); ++it) {
		string occupationLine = it->second;
		for(int i = 0; i < occupationLine.length(); ++i)
			if(occupationLine.at(i) == c)
				++cnt;
	}
	return cnt;
}

int ParkingDatasetDescription::get_no_empty_spots() {
	return count_occupation('e');
}

int ParkingDatasetDescription::get_no_occupied_spots() {
	return count_occupation('o');
}

