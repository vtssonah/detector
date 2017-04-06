#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <string>
#include <fstream>
#include <ctime>
#include <pthread.h>
#include "util.hpp"
#include "ParkingDatasetDescription.h"
#include "ParkingClassifier.h"
#include "SVMParkingClassifier.h"
#include "HarrisPC.h"
#include "NaiveGaussianHOGPC.h"

using namespace std;
using namespace cv;
using namespace cv::ml;


#define TRAINED_SVM "vehicle_detector.yml"
#define PKLOT_SVM "PKLot_vehicle_detector.yml"
#define INPUT_SVM PKLOT_SVM
#define WINDOW_NAME "Marked cars."
#define	IMAGE_SIZE Size(32,32)

static int no_spots = 1;

// /home/timo/ImageData/analysis/Occupied2 /home/timo/ImageData/analysis/Empty2 PKLot_vehicle_detector.yml
// /home/timo/ImageData/DataAquisition/PDI/PDI1484208758/1484150110193.jpg
// /home/timo/ImageData/DataAquisition/mon.sonah.xyz/ASAP_10.121.0.100/monitoring/cn0/2017-02-13_07:53:15_cn0.jpg

Ptr<TrainData> convert_train_data(Mat train_data, vector<int> labels) {
	return TrainData::create(train_data, ROW_SAMPLE, labels);
}

Ptr<SVM> automatic_training(Mat train_data, vector<int> labels) {
	Ptr<TrainData> data = convert_train_data(train_data, labels);

}

Ptr<TrainData> load_train_data() {
}

//Ptr<SVM> train_svm_hog(string path_positive_samples, string path_negative_samples) {
//	vector< Mat > pos_lst;
//	vector< Mat > full_neg_lst;
//	vector< Mat > neg_lst;///home/timo/ImageData/analysis/Empty /home/timo/ImageData/analysis/Occupied PKLot_vehicle_detector3.yml
//	vector< Mat > gradient_lst;
//	vector< int > labels;
//	load_images(path_positive_samples, pos_lst, IMAGE_SIZE, 3000);
//	labels.assign(pos_lst.size(), +1);
//	cout << "loeaded number of p: " << pos_lst.size() << endl;
/////home/timo/ImageData/PKLot/PKLotSegmented/Empty /home/timo/ImageData/PKLot/PKLotSegmented/Occupied PKLot_vehicle_detector3.yml
//	load_images(path_negative_samples, full_neg_lst, IMAGE_SIZE, 3000);
//	labels.insert(labels.end(), full_neg_lst.size(), -1);
//	cout << "loeaded number of p: " << full_neg_lst.size() << endl;
//
//	compute_hog(pos_lst, gradient_lst, IMAGE_SIZE);
////	cout << pos_lst.size() << endl;
//	compute_hog(full_neg_lst, gradient_lst, IMAGE_SIZE);
//
//	/* Default values to train SVM */
//	Ptr<SVM> svm = SVM::create();
//	svm->setCoef0(0.0);
//	svm->setDegree(3);
//	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-3));
//	svm->setGamma(0.5);
//	svm->setKernel(SVM::LINEAR);
//	svm->setNu(0.5);
//	svm->setP(0.1); // for EPSILON_SVR, epsilon in loss function?
//	svm->setC(0.01); // From paper, soft classifier
//	svm->setType(SVM::EPS_SVR); // C_SVC; // EPSILON_SVR; // may be also NU_SVR; // do regression task
//
//	Mat train_data;
//	cout << "number of gradients: " << gradient_lst.size() << endl;
//	cout << "number of samples  : " << labels.size() << endl;
//	convert_to_ml(gradient_lst, train_data);
//
////	clog << "Start training...";
//	svm->train(train_data, ROW_SAMPLE, Mat(labels));
//	return svm;
//}

//Ptr<SVM> train_svm_on_dataset(vector<Mat>& pos_list, vector<Mat>& neg_list) {
//	vector< Mat > gradient_lst;
//	vector< int > labels;
//	labels.assign(pos_list.size(), +1);
//	labels.insert(labels.end(), neg_list.size(), -1);
//	compute_hog(pos_list, gradient_lst, IMAGE_SIZE);
////	cout << pos_lst.size() << endl;
//	compute_hog(neg_list, gradient_lst, IMAGE_SIZE);
//
//	/* Default values to train SVM */
//	Ptr<SVM> svm = SVM::create();
//	svm->setCoef0(0.0);
//	svm->setDegree(3);
//	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-3));
//	svm->setGamma(0.5);
//	svm->setKernel(SVM::LINEAR);
//	svm->setNu(0.5);
//	svm->setP(0.1); // for EPSILON_SVR, epsilon in loss function?
//	svm->setC(0.01); // From paper, soft classifier
//	svm->setType(SVM::EPS_SVR); // C_SVC; // EPSILON_SVR; // may be also NU_SVR; // do regression task
//
//	Mat train_data;
////	cout << "moin" << endl;
//	convert_to_ml(gradient_lst, train_data);
//
////	clog << "Start training...";
//	svm->train(train_data, ROW_SAMPLE, Mat(labels));
//	return svm;
//}


void partitionData(vector<Mat> input_files, vector< vector<Mat> >& partition, int no_files_per_partition) {
	vector<Mat> cPartition;// current partition.
	for(int i = 0; i < input_files.size(); ++i) {
		cPartition.push_back(input_files[i]);
		if(cPartition.size() >= no_files_per_partition) {
			partition.push_back(cPartition);
			cPartition = vector<Mat>();
		}
	}
}

void partitionDataBatch(vector<Mat> input_files, vector< vector<Mat> >& partition, int no_partitions) {
	partitionData(input_files, partition, input_files.size() / no_partitions);
}

void experiment_testing_suite() {
	ParkingDatasetDescription pdd = ParkingDatasetDescription::load("/home/timo/ImageData/DataAquisition/mon.sonah.xyz/ASAP_10.121.0.103/monitoring/labeling.lbl");
	class DummyClassifier : public ParkingClassifier {
	public:
		DummyClassifier() : ParkingClassifier::ParkingClassifier() {}
		virtual ~DummyClassifier() {}
		string classify(Mat image) {
			string res;
			for(int i=0;i < this->parking_spots.size(); ++i) {
				res.append("e");
			}
			return res;
		}
	} dummy;
	vector<int> scores = dummy.test(pdd);
	for(int i = 0; i < scores.size(); ++i)
		cout << scores[i] << endl;
}

void addAllExcept(vector<Mat>& src, vector<Mat>& dst, int index) {
	for(int i=0; i < index; ++i)
		dst.push_back(src[i]);
	for(int i=index+1; i < src.size(); ++i)
		dst.push_back(src[i]);
}

double get_accuracy(vector<int> v) {
	int correct = v[0] + v[3];
	return (double)correct/(correct+v[1]+v[2]);
}

static ParkingDatasetDescription *pdd;

struct thread_data {
   SVM_ParkingClassifier *classifier;
   string filename;
};


void *cross_validate(void* args) {// Parameter: Pointer to SVM, do testing here.
	struct thread_data *my_data;
	my_data = (struct thread_data *) args;
	SVM_ParkingClassifier *classifier = my_data->classifier;
	vector<int> score = classifier->test(*pdd);
	double candidate_accuracy = get_accuracy(score);
	cout << "  Accuracy: " << candidate_accuracy << " (" << my_data->filename << ")" << endl;
	delete my_data->classifier;
	delete my_data;
	pthread_exit(NULL);
}

void experiment_training_set_reduction(string path_to_positive_samples, string path_to_negative_samples) {
	ParkingDatasetDescription pdd2 = ParkingDatasetDescription::load("/home/timo/ImageData/DataAquisition/data_cirratic/cirrantic.lbl");
	pdd = &pdd2;
	vector< Mat > image_list_positive;
	vector< vector<Mat> > partition_positive;
	vector< Mat > image_list_negative;
	vector< vector<Mat> > partition_negative;
	vector< string > image_names_positive;
	vector< string > image_names_negative;
	load_images_with_names(path_to_positive_samples, image_list_positive, image_names_positive, IMAGE_SIZE,0);
	load_images_with_names(path_to_negative_samples, image_list_negative, image_names_negative, IMAGE_SIZE,0);
	SVM_ParkingClassifier svm_classifier(IMAGE_SIZE, true);
	svm_classifier.set_parking_spots(pdd->get_parking_spots());

	// Initially set the
	Ptr<SVM> bestSVM = train_hog_svm_on_dataset(image_list_positive, image_list_negative, IMAGE_SIZE);
	svm_classifier.set_SVM(bestSVM);
	bestSVM->save("Guelle.yml");

	vector<int> bestScore = svm_classifier.test(pdd2);
	double best_accuracy = get_accuracy(bestScore);
	cout << "Dataset (empty: " << pdd2.get_no_empty_spots() << "\toccupied: " << pdd2.get_no_occupied_spots() << ")" << endl;
	cout << "Initial accuracy: " << best_accuracy << endl;
	// Leave-one-out for both the positive and negative samples.
//	pthread_t thread[10];
	int iBest_positive = 0;
	int iBest_negative = 0;
	for(int i = 0; i < image_list_negative.size(); ++i) {
		vector<Mat> train_images;
		addAllExcept(image_list_negative, train_images, i);
		Ptr<SVM> candidateSVM = train_hog_svm_on_dataset(image_list_positive, train_images, IMAGE_SIZE);
		SVM_ParkingClassifier *tmp_svm_classifier = new SVM_ParkingClassifier(IMAGE_SIZE, false);
		tmp_svm_classifier->set_parking_spots(pdd2.get_parking_spots());
		tmp_svm_classifier->set_SVM(candidateSVM);

//		struct thread_data *my_data = new thread_data;
//		my_data->classifier = tmp_svm_classifier;
//		my_data->filename = image_names_negative[i];
//		cout << "creating thread" << endl;
//		pthread_create(&thread[i], NULL, cross_validate, my_data);
//		if(i==2) {
//			cout << "Joining threads.." << endl;
//			int *ret;
//			for(int j = 0; j <=i; ++j) pthread_join(thread[j], (void**)&ret);
//			return;
//		}
		vector<int> candidateScore = svm_classifier.test(pdd2);
		double candidate_accuracy = get_accuracy(candidateScore);
		cout << "  Accuracy: " << candidate_accuracy << "\tvs. " << best_accuracy << " (" << image_names_negative[i] << ")" << endl;
		if(best_accuracy < candidate_accuracy) {
			iBest_negative = i;
			bestSVM = candidateSVM;
			bestScore = candidateScore;
			best_accuracy = candidate_accuracy;
			cout << "found a new best score, dropping a negative example!" << endl;
		}
	}
//	for(int i = 0; i < image_list_positive.size(); ++i) {
//		vector<Mat> train_images;
//		addAllExcept(image_list_positive, train_images, i);
//		Ptr<SVM> candidateSVM = train_hog_svm_on_dataset(train_images, image_list_negative, IMAGE_SIZE);
//		svm_classifier.set_SVM(candidateSVM);
//		vector<int> candidateScore = svm_classifier.test(pdd2);
//		double candidate_accuracy = get_accuracy(candidateScore);
//		cout << "  Accuracy: " << candidate_accuracy << "\tvs. " << best_accuracy << " (" << image_names_positive[i] << ")" << endl;
//		if(best_accuracy < candidate_accuracy) {
//			iBest_positive = i;
//			bestSVM = candidateSVM;
//			bestScore = candidateScore;
//			best_accuracy = candidate_accuracy;
//			cout << "found a new best score, dropping a positive example!" << endl;
//		}
//	}
	bestSVM->save("Best_SVM.yml");
	FILE* log = fopen("log.txt", "wb");
	fseek(log, 0, SEEK_END);
//	fwrite((image_names_positive[iBest_positive]+"\n").c_str(), 1, image_names_positive[iBest_positive].length()+1, log);
	fwrite((image_names_negative[iBest_negative]+"\n").c_str(), 1, image_names_negative[iBest_negative].length()+1, log);
	fclose(log);
}

int histogram_svm_training(int argc, char** argv) {
	if(argc > 3) {
		string correctedPath1 = argv[1];
		if(correctedPath1.at(correctedPath1.length()-1) != '/')
			correctedPath1.append("/");
		string correctedPath2 = argv[2];
		if(correctedPath2.at(correctedPath2.length()-1) != '/')
			correctedPath2.append("/");
		train_histogram_svm_and_store_in_file(argv[1],argv[2],argv[3]);
		return 0;
	} else {
		cout << "USAGE:" << endl;//TODO: extend this.
		return 1;
	}

}

int hog_svm_training(int argc, char** argv) {
	if(argc > 3) {
		string correctedPath1 = argv[1];
		if(correctedPath1.at(correctedPath1.length()-1) != '/')
			correctedPath1.append("/");
		string correctedPath2 = argv[2];
		if(correctedPath2.at(correctedPath2.length()-1) != '/')
			correctedPath2.append("/");
		train_hog_svm_and_store_in_file(argv[1],argv[2],argv[3]);
		return 0;
	} else {
		cout << "USAGE:" << endl;//TODO: extend this.
		return 1;
	}

}

int test_svm_on_dataset(int argc, char** argv) {
	if(argc > 2) {
		int i_svm = 1;
		string arg1(argv[1]);
		if(arg1=="--show-detections") {
			if(argc < 3)
				return 1;
			i_svm = 2;
		}
		Ptr<SVM> svm = StatModel::load<SVM>(argv[i_svm]);
		ParkingDatasetDescription pdd = ParkingDatasetDescription::load(argv[i_svm+1]);
		SVM_ParkingClassifier svm_classifier(IMAGE_SIZE, true);
		svm_classifier.set_parking_spots(pdd.get_parking_spots());
		svm_classifier.set_SVM(svm);
		svm_classifier.set_output_path("/home/timo/output");
		svm_classifier.set_show_progress(true);
		vector<int> score = svm_classifier.test(pdd);
		if(i_svm == 2) {
			vector<string> image_list = get_keys(pdd.get_occupation());
			vector<Mat> out_images = svm_classifier.get_detection_images();
			for(int i = 0; i < image_list.size(); ++i) {
				imwrite("o" + image_list[i], out_images[i]);
			}
		}
		cout << " true label | classified | count" << endl;
		cout << " false      | false      | " << score[0] << endl;
		cout << " false      | true       | " << score[1] << "\t(false positives)" << endl;
		cout << " true       | false      | " << score[2] << "\t(false negatives)" << endl;
		cout << " true       | true       | " << score[3] << endl;
		cout << "   Accuracy: " << get_accuracy(score) << endl;
		return 0;
	} else {
		cout << "USAGE: " << argv[0] << " PATH_TO_SVM PATH_TO_PARKING_DATASET_DESCRIPTION" << endl;
		return 1;
	}
	return 0;
}

int test_harris(int argc, char** argv) {
	HarrisPC pc = HarrisPC(Size(0,0));
	ParkingDatasetDescription pdd = ParkingDatasetDescription::load(argv[3]);
	pc.set_show_progress(false);
	pc.set_parking_spots(pdd.get_parking_spots());
	int best_t = 0;
	pc.set_threshold(best_t);
	pc.set_output_path("/home/timo/output");
	vector<int> best_score = pc.test(pdd);
	for(int t = 10; t < 100; t+=10) {
		pc.set_threshold(t);
		vector<int> score = pc.test(pdd);
		if(get_accuracy(score) > get_accuracy(best_score)) {
			best_score = score;
			best_t = t;
		}
	}
	cout << " true label | classified | count" << endl;
	cout << " false      | false      | " << best_score[0] << endl;
	cout << " false      | true       | " << best_score[1] << "\t(false positives)" << endl;
	cout << " true       | false      | " << best_score[2] << "\t(false negatives)" << endl;
	cout << " true       | true       | " << best_score[3] << endl;
	cout << "   Accuracy (@" << best_t << "): " << get_accuracy(best_score) << endl;

	return 0;
}

int analyze_naive_gaussian(int argc, char** argv) {
	// Calculate the HOG features for each training sample and show the result.
	string path_to_positive_samples = argv[1];
	if(path_to_positive_samples.at(path_to_positive_samples.length()-1) != '/')
		path_to_positive_samples.append("/");
	vector< Mat > image_list_positive;
	vector< string > image_names_positive;
	load_images_with_names(path_to_positive_samples, image_list_positive, image_names_positive, IMAGE_SIZE,0);
	vector<Mat> hog_values;
//	for(vector<Mat>::iterator it = image_list_positive.begin(); it != image_list_positive.end(); ++it) {
		compute_hog(image_list_positive, hog_values, Size(40,40));
//	}
	vector<float> means, vars;
	get_mean_var_vector(hog_values, means, vars);
	cout << means.size() << endl;
	vector<int> mask;
	for(int i=0;i<means.size();++i) {
		int smaller1 = 0.015625 <= vars[i] && vars[i] <= 1.0;
		if(smaller1)
			cout << means[i] << "," << vars[i] << ";";
		else
			cout << "0,0;";
		mask.push_back(smaller1 && vars[i] != 0);
	}
	cout << endl;
	cout << "  TEST" << endl;
	NaiveGaussianHOG_PC naiveGaussian(means, vars, mask);
	string path_to_test_samples = argv[2];
	if(path_to_test_samples.at(path_to_test_samples.length()-1) != '/')
		path_to_test_samples.append("/");
	vector< Mat > image_list_test, hog_values_test;
	vector< string > image_names_test;
	load_images_with_names(path_to_test_samples, image_list_test, image_names_test, IMAGE_SIZE,0);
	compute_hog(image_list_test, hog_values_test, Size(40,40));
	for(int i=0;i<image_list_test.size();++i){
		cout << image_names_test[i] << ": " << naiveGaussian.isObject(hog_values_test[i]) << endl;
//		for(int j=0;j<hog_values_test[i])
	}
//	cout << Mat(vars).t() << endl;
	return 0;
}

int train_histogram_svm_and_test(int argc, char** argv) {
	// The first 3 parameters are the training set and the output file name of the svm.
	// Then comes an optional parameter (--show-detections) and the testing set.
	if(argc > 4) {
		string correctedPath1 = argv[1];
		if(correctedPath1.at(correctedPath1.length()-1) != '/')
			correctedPath1.append("/");
		string correctedPath2 = argv[2];
		if(correctedPath2.at(correctedPath2.length()-1) != '/')
			correctedPath2.append("/");
		histogram_svm_training(argc, argv);
		string svm_filename = argv[3];
		string dataset_filename = argv[4];
		char* args_for_test[4];
		args_for_test[1] = argv[4];
		args_for_test[2] = argv[5];
		int no_args_for_test = 3;
		if(dataset_filename == "--show-detections") {
			args_for_test[3] = args_for_test[2];
			args_for_test[2] = argv[3];
			no_args_for_test = 4;
		}
		cout << "Training done. Testing.." << endl;
		test_svm_on_dataset(no_args_for_test, args_for_test);

		return 0;
	}
	return 1;
}

int train_hog_svm_and_test(int argc, char** argv) {
	// The first 3 parameters are the training set and the output file name of the svm.
	// Then comes an optional parameter (--show-detections) and the testing set.
	if(argc > 4) {
		string correctedPath1 = argv[1];
		if(correctedPath1.at(correctedPath1.length()-1) != '/')
			correctedPath1.append("/");
		string correctedPath2 = argv[2];
		if(correctedPath2.at(correctedPath2.length()-1) != '/')
			correctedPath2.append("/");
		hog_svm_training(argc, argv);
		string svm_filename = argv[3];
		string dataset_filename = argv[4];
		char* args_for_test[4];
		args_for_test[1] = argv[4];
		args_for_test[2] = argv[5];
		int no_args_for_test = 3;
		if(dataset_filename == "--show-detections") {
			args_for_test[3] = args_for_test[2];
			args_for_test[2] = argv[3];
			no_args_for_test = 4;
		}
		cout << "Training done. Testing.." << endl;
		test_svm_on_dataset(no_args_for_test, args_for_test);

		return 0;
	}
	return 1;
}

const string PATH_POSITIVE_SAMPLES="/home/timo/ImageData/analysis/Occupied2/";
const string PATH_NEGATIVE_SAMPLES="/home/timo/ImageData/analysis/Empty3/";


void abc() {
	Mat m(Size(3,3), CV_32F);
	m.at<float>(0, 0) = 3.0f;// indexed as row, column
	m.at<float>(1, 0) = 4.0f;
	m.at<float>(2, 0) = 15.0f;

	m.at<float>(0, 1) = 4.0f;
	m.at<float>(1, 1) = 84.0f;
	m.at<float>(2, 1) = 22.0f;

	m.at<float>(0, 2) = 59.0f;
	m.at<float>(1, 2) = 4.0f;
	m.at<float>(2, 2) = 1.0f;
	cout << m << endl;
	double minVal, maxVal;
	Point minLoc, maxLoc;
	minMaxLoc(m, &minVal, &maxVal, &minLoc, &maxLoc);
	cout << minVal << "\t" << maxVal << endl;
	minMaxLoc(m.row(0), &minVal, &maxVal, &minLoc, &maxLoc);
	cout << minVal << "\t" << maxVal << endl;
	Mat meanMat, stddevMat;
	meanStdDev(m.row(0), meanMat, stddevMat);
	cout << meanMat << endl;
	cout << stddevMat << endl;
	vector<float> means, vars;
	get_mean_var_vector(m, means, vars);
	for(int i=0;i<means.size();++i){
		cout << means[i] << endl;
		cout << vars[i] << endl;
	}
	cout << endl;

	vector<float> v1;v1.push_back(3);v1.push_back(4);v1.push_back(59);
	vector<float> v2;v2.push_back(4);v2.push_back(84);v2.push_back(4);
	vector<float> v3;v3.push_back(15);v3.push_back(22);v3.push_back(1);
	Mat m_v1(v1);// m_v1 becomes a one-column vector.
	cout << m_v1 << endl;
	vector<Mat> vMat;
	vMat.push_back(m_v1);
	vMat.push_back(Mat(v2));
	vMat.push_back(Mat(v3));
	means.clear();vars.clear();
	get_mean_var_vector(vMat, means, vars);
	for(int i=0;i<means.size();++i){
		cout << means[i] << endl;
		cout << vars[i] << endl;
	}
}

int main2(int argc, char** argv) {
//	cv::Sobel(src, dst, ddepth, dx, dy, ksize, scale, delta, borderType)
	if(argc > 2) {
		Mat image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
		vector<Mat> img_list;
		img_list.push_back(image);

		Mat bgr[3];
		split(image, bgr);
//		cout << bgr[0] << endl;
//		cout << bgr[1] << endl;
//		cout << bgr[2] << endl;
		imwrite("blue.png",bgr[0]); //blue channel
		imwrite("green.png",bgr[1]); //green channel
		imwrite("red.png",bgr[2]); //red channel

		vector<Mat> gradient_lst;
		compute_histogram(img_list, gradient_lst, Size(0,0));
		// read the parking areas.
		vector<Parkplatz*> parkingspots;
		vector<Mat> transformationMat;
		for(int p = 2; p < argc; ++p) {
			Parkplatz* parkplatz = Parkplatz::fromString2(argv[p]);
			parkingspots.push_back(parkplatz);
		}
		HarrisPC pc = HarrisPC(Size(0,0));
		pc.set_parking_spots(parkingspots);
		pc.set_threshold(80);
		cout << pc.classify(image);
		return 0;
	}
	return 0;
}

// /home/timo/ImageData/DataAquisition/mon.sonah.xyz/ASAP_10.121.0.100/monitoring/cn0/100cn0.lbl
int main(int argc, char** argv) {
//	return analyze_naive_gaussian(argc, argv);
//	abc();
//	return 0;
	return train_histogram_svm_and_test(argc, argv);
	return train_hog_svm_and_test(argc, argv);
//	return hog_svm_training(argc, argv);
	return test_svm_on_dataset(argc, argv);
//	cout << read_file("/home/timo/ImageData/DataAquisition/mon.sonah.xyz/ASAP_10.121.0.103/monitoring/labeling.lbl");
//	experiment_testing_suite();
//	return test_harris(argc, argv);
	experiment_training_set_reduction("/home/timo/Desktop/cars/positives/wheel/", "/home/timo/Desktop/cars/negatives/wheel/");
	return 0;
//	ParkingDatasetDescription pdd = ParkingDatasetDescription::load("/home/timo/ImageData/DataAquisition/mon.sonah.xyz/ASAP_10.121.0.103/monitoring/labeling.lbl");
//	Ptr<TrainData> train_data = pdd.to_TrainData();
//	no_spots = pdd.get_parking_spots().size();
//	Mat scores = test_binary_classifier(train_data, simple_classifier);
////	vector<Point2f> pts;
////	pts.push_back(Point(10,20));
////	pts.push_back(Point(25,20));
////	pts.push_back(Point(25,35));
////	pts.push_back(Point(10,35));
////	Rect r = boundingRect(Mat(pts));
////	cout << r << endl;
//	Ptr<TrainData> trainData = pdd.get_HOG_features(IMAGE_SIZE);
//	Ptr<SVM> svm = SVM::create();
//	svm->setCoef0(0.0);
//	svm->setDegree(3);
//	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-3));
//	svm->setGamma(0.5);
//	svm->setKernel(SVM::LINEAR);
//	svm->setNu(0.5);
//	svm->setP(0.1); // for EPSILON_SVR, epsilon in loss function?
//	svm->setC(0.01); // From paper, soft classifier
//	svm->setType(SVM::EPS_SVR); // C_SVC; // EPSILON_SVR; // may be also NU_SVR; // do regression task
//	svm->trainAuto(trainData, 20);
//	svm->save(argv[3]);
////	return 0;
////	cout << trainData->getResponses() << endl;
//	cout << pdd.get_directory() << endl;
//	vector<Parkplatz*> p = pdd.get_parking_spots();
//	map<string, string> occupation = pdd.get_occupation();
//	cout << occupation.size() << endl;
//	vector<string> keys = get_keys(occupation);
//	cout << keys.size();
//	for(vector<string>::iterator it = keys.begin(); it != keys.end(); ++it) {
//		cout << (*it) << endl;
//	}
	if(argc > 3) {//Parameters: /home/timo/ImageData/PKLot/PKLotSegmented/Occupied /home/timo/ImageData/PKLot/PKLotSegmented/Empty PKLot_vehicle_detector.yml
		string correctedPath1 = argv[1];
//		cout << "moin" << endl;
		if(correctedPath1.at(correctedPath1.length()-1) != '/')
			correctedPath1.append("/");
		string correctedPath2 = argv[2];
		if(correctedPath2.at(correctedPath2.length()-1) != '/')
			correctedPath2.append("/");
//		cout << "Training SVM.. " << endl;
		vector< Mat > image_list_positive;
		vector< vector<Mat> > partition_positive;
		vector< Mat > image_list_negative;
		vector< vector<Mat> > partition_negative;
		load_images(correctedPath1, image_list_positive, IMAGE_SIZE, 3000);
		load_images(correctedPath2, image_list_negative, IMAGE_SIZE, 3000);
		partitionData(image_list_positive, partition_positive, 80);
		partitionData(image_list_negative, partition_negative, 68);
//		cout << image_list_negative.size() << endl;
		Ptr<SVM> svm = train_hog_svm_on_dataset(image_list_positive, image_list_negative, IMAGE_SIZE);
		string outFilename = argv[3];
//		for(int i = 0; i < partition_negative.size(); ++i) {
////			cout << partition_positive[i].size() << endl;
//			Ptr<SVM> svm = train_svm_on_dataset(image_list_positive, partition_negative[i]);
			string name = outFilename;
//			char buf[32];
//			sprintf(buf, "%d", i);
//			name.append(buf);
			svm->save(name);
//		}
//		cout << "done." << argv[3] << endl;
//		svm->save(argv[3]);
	} else if(argc > 1) {
		Ptr<SVM> svm;
		HOGDescriptor hog;
		hog.winSize = IMAGE_SIZE;
		vector< Rect > locations;
		svm = StatModel::load<SVM>(INPUT_SVM);
		// Set the trained svm to my_hog
		vector< float > hog_detector;
		get_svm_detector(svm, hog_detector);
		hog.setSVMDetector(hog_detector);
		Mat img = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
		GaussianBlur(img, img, Size(7,7), 1);
		hog.detectMultiScale(img, locations);
		Mat draw = img.clone();
		draw_locations(draw, locations, Scalar(0, 255, 0));
		int num_of_vehicles=0;
		cout << locations.size() << endl;
		for(int i = 0; i < locations.size(); ++i) {
			Rect r = locations[i];
			cout << locations[i] << endl;

			// Center point of the vehicle
			Point center(r.x + r.width / 2, r.y + r.height / 2);

//			if (abs(center.y - img.rows * 2 / 3) < 2) {
				++num_of_vehicles;
//				line(draw, Point(0, img.rows * 2 / 3), Point(img.cols / 2, img.rows * 2 / 3), Scalar(0, 255, 0), 3);
				imshow(WINDOW_NAME, draw);
//				waitKey(500);
//			}
//			else
//				line(draw, Point(0, img.rows * 2 / 3), Point(img.cols / 2, img.rows * 2 / 3), Scalar(0, 0, 255), 3);
		}
		cout << "Number of vehicles: " << num_of_vehicles << endl;
		imshow(WINDOW_NAME, draw);
		waitKey();
	}
	return 0;
}
