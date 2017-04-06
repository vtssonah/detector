/**
 * Sonah utility functions.
 * @author Timo Hinrichs (hinrichs@sonah-parking.com)
 * @version 0.02
 */
#ifndef UTIL_HPP_
#define UTIL_HPP_

#include <vector>
#include <string>
#include <opencv/cv.h>
#include <opencv2/ml.hpp>
#include <opencv2/objdetect.hpp>
#include "Parkplatz.hpp"

using namespace std;
using namespace cv;
using namespace cv::ml;

vector<string> ssplit(string str, char delimiter);
string read_file(const char* filename);
vector<int> getPeakIndices(Mat histogram);
vector<Mat> getTransformedParkingSpots(Mat image, vector<Parkplatz*> parkingSpots, int width=32, int height=64);
/**
 * Extracts the hog-svm detector from the SVM.
 */
void get_svm_detector(const Ptr<SVM>& svm, vector< float > & hog_detector);
void draw_locations(Mat & img, const vector< Rect > & locations, const Scalar & color);
/**
 * TODO: Explain what this method does!
 */
void convert_to_ml(const std::vector< cv::Mat > & train_samples, cv::Mat& trainData);
void load_images(string directory, vector<Mat>& image_list, Size image_size, int max_no_images=0);
void load_images_with_names(string directory, vector<Mat>& image_list, vector<string>& image_name_list, Size image_size, int max_no_images=0);
vector<string> files_in_directory(string directory, string suffix);
vector<string> files_in_directory(string directory, vector<string> suffix);
void compute_histogram(const vector< Mat > & img_lst, vector< Mat > & gradient_lst, const Size & size);
void compute_hog(const vector< Mat > & img_lst, vector< Mat > & gradient_lst, const Size & size);
Ptr<SVM> train_hog_svm(const vector< Mat > & gradient_lst, const vector< int > & labels);
Ptr<SVM> train_histogram_svm_on_dataset(vector<Mat>& pos_list, vector<Mat>& neg_list, Size image_size);
Ptr<SVM> train_hog_svm_on_dataset(vector<Mat>& pos_list, vector<Mat>& neg_list, Size image_size);
void train_hog_svm_and_store_in_file(string path_to_positive_samples, string path_to_negative_samples, string outFile);
void train_histogram_svm_and_store_in_file(string path_to_positive_samples, string path_to_negative_samples, string outFile);

vector<string> get_keys(map<string, string> m);
Mat get_hogdescriptor_visu(const Mat& color_origImg, vector<float>& descriptorValues, const Size & size);
void get_difference_vector(vector<float> v1, vector<float> v2, vector<float>& res);
double get_difference_sum(vector<float> v1, vector<float> v2);
void get_mean_var_vector(Mat m, vector<float>& means, vector<float>& vars);
void get_mean_var_vector(vector< Mat >& m, vector<float>& means, vector<float>& vars);
void get_mean_var_vector(vector< vector<float> >& m, vector<float>& means, vector<float>& vars);

#endif
