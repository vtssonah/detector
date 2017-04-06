#include <sstream>
#include <iostream>
#include <dirent.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <malloc.h>
#include "util.hpp"

using namespace std;

vector<string> ssplit(string str, char delimiter) {
	vector<string> internal;
	stringstream ss(str);
	string tok;

	while(getline(ss, tok, delimiter)) {
		internal.push_back(tok);
	}

	return internal;
}

string read_file(const char* filename) {
	FILE *f = fopen(filename, "rb");
	fseek(f, 0, SEEK_END);
	int length = ftell(f);
	char *buf = (char*)malloc(length);
	fseek(f, 0, SEEK_SET);
	fread(buf, 1, length, f);
	fclose(f);
	string res(buf);
	free(buf);
	return res;
}

vector<int> getPeakIndices(cv::Mat histogram) {
	vector<int> res;
	int ascending = 0;
	int last = (int)histogram.at<float>(0, 0);
//	std::cout << last << std::endl;
//	std::cout << histogram.rows<< std::endl;
	for(int i=1; i < histogram.rows; ++i) {
//		std::cout << (int)histogram.at<float>(i,0) << std::endl;
		int count = (int)histogram.at<float>(i,0);
		if(ascending && count < last)
			res.push_back(i);
		ascending = (last <= count);
		last = count;
	}
	// handle the last index as a special case.
	return res;
}

vector<cv::Mat> getTransformedParkingSpots(cv::Mat image, vector<Parkplatz*> parkingSpots, int width, int height) {
	vector<cv::Mat> res;
	vector<cv::Point2f> dstPoints;
	dstPoints.push_back(cv::Point2f(0,0));
	dstPoints.push_back(cv::Point2f(0,height));
	dstPoints.push_back(cv::Point2f(width,height));
	dstPoints.push_back(cv::Point2f(width,0));
	cv::Mat dstMat(dstPoints);
	for(int i = 0; i < parkingSpots.size(); ++i) {
		cv::Mat boundary;
		parkingSpots[i]->getBoundary(&boundary);
		// do perspective transform and scale to width x height.
		cv::Mat warpedParkingSpot;
		cv::warpPerspective(image, warpedParkingSpot, cv::getPerspectiveTransform(boundary, dstMat), cv::Size(width,height));
		res.push_back(warpedParkingSpot);
	}
	return res;
}

void get_svm_detector(const Ptr<SVM>& svm, vector< float > & hog_detector) {
	// get the support vectors
	Mat sv = svm->getSupportVectors();
	const int sv_total = sv.rows;
	// get the decision function
	Mat alpha, svidx;
	double rho = svm->getDecisionFunction(0, alpha, svidx);

//	CV_Assert(alpha.total() == 1 && svidx.total() == 1 && sv_total == 1);
//	CV_Assert((alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
//		(alpha.type() == CV_32F && alpha.at<float>(0) == 1.f));
//	CV_Assert(sv.type() == CV_32F);
	hog_detector.clear();

	hog_detector.resize(sv.cols + 1);
	memcpy(&hog_detector[0], sv.ptr(), sv.cols*sizeof(hog_detector[0]));
	hog_detector[sv.cols] = (float)-rho;
}

void draw_locations(Mat & img, const vector< Rect > & locations, const Scalar & color)
{
	if (!locations.empty())
	{
		vector< Rect >::const_iterator loc = locations.begin();
		vector< Rect >::const_iterator end = locations.end();
		for (; loc != end; ++loc)
		{
			rectangle(img, *loc, color, 2);
		}
	}
}

void convert_to_ml(const std::vector< cv::Mat > & train_samples, cv::Mat& trainData)
{
	//--Convert data
	const int rows = (int)train_samples.size();
//	cout << "convert_to_ml" << train_samples.size() << endl;
	const int cols = (int)std::max(train_samples[0].cols, train_samples[0].rows);
	cv::Mat tmp(1, cols, CV_32FC1); //< used for transposition if needed
	trainData = cv::Mat(rows, cols, CV_32FC1);
	vector< Mat >::const_iterator itr = train_samples.begin();
	vector< Mat >::const_iterator end = train_samples.end();
	for (int i = 0; itr != end; ++itr, ++i)
	{
		CV_Assert(itr->cols == 1 ||
			itr->rows == 1);
		if (itr->cols == 1)
		{
			transpose(*(itr), tmp);
			tmp.copyTo(trainData.row(i));
		}
		else if (itr->rows == 1)
		{
			itr->copyTo(trainData.row(i));
		}
	}
}

void load_images(string directory, vector<Mat>& image_list, Size image_size, int max_no_images) {
	Mat img;
	vector<string> files;
	files = files_in_directory(directory, ".jpg");
	if(max_no_images <= 0)
		max_no_images = files.size();
//	cout << "files.size()=" << files.size() << endl;

	for (int i = 0; i < files.size() && i < max_no_images; ++i) {
//		cout << files.at(i) << endl;
		img = imread(files.at(i));
		if (img.empty()) {
//			cout << "image is empty" << endl;
			continue;
		}
#ifdef _DEBUG
		imshow("image", img);
		waitKey(10);
#endif
		resize(img, img, image_size);
		image_list.push_back(img.clone());
	}
}

void load_images_with_names(string directory, vector<Mat>& image_list, vector<string>& image_name_list, Size image_size, int max_no_images) {
	Mat img;
	vector<string> files;
	vector<string> suffixes;
	suffixes.push_back(".jpg");
	suffixes.push_back(".png");
	files = files_in_directory(directory, suffixes);
	if(max_no_images <= 0)
		max_no_images = files.size();
//	cout << "files.size()=" << files.size() << endl;

	for (int i = 0; i < files.size() && i < max_no_images; ++i) {
//		cout << files.at(i) << endl;
		img = imread(files.at(i), CV_LOAD_IMAGE_COLOR);
		if (img.empty()) {
//			cout << "image is empty" << endl;
			continue;
		}
#ifdef _DEBUG
		imshow("image", img);
		waitKey(10);
#endif
		resize(img, img, image_size);
		image_list.push_back(img.clone());
		image_name_list.push_back(files.at(i));
	}
}

vector<string> files_in_directory(string directory, string suffix)
{
	vector<string> files;
	DIR* dir = opendir(directory.c_str());
	struct dirent *epdf;
	while (((epdf = readdir(dir)) != NULL)) {
		string filename = epdf->d_name;
		if(filename.length() >= suffix.length() && filename.substr(filename.length()-suffix.length(), suffix.length()) == suffix) {
			string pathAndFile(directory);
			pathAndFile.append(epdf->d_name);
			files.push_back(pathAndFile);
		}
	}
//	char buf[256];
//	string command;
//
//#ifdef __linux__
//	command = "ls " + directory;
//	shared_ptr<FILE> pipe(popen(command.c_str(), "r"), pclose);
//
//	char cwd[256];
//	getcwd(cwd, sizeof(cwd));
//
//	while (!feof(pipe.get()))
//		if (fgets(buf, 256, pipe.get()) != NULL) {
//			string file(cwd);
//			file.append("/");
//			file.append(buf);
//			file.pop_back();
//			files.push_back(file);
//		}
//#else
//	command = "dir /b /s " + directory;
//	FILE* pipe = NULL;
//
//	if (pipe = _popen(command.c_str(), "rt"))
//		while (!feof(pipe))
//			if (fgets(buf, 256, pipe) != NULL) {
//				string file(buf);
//				file.pop_back();
//				files.push_back(file);
//			}
//	_pclose(pipe);
//#endif

	return files;
}

vector<string> files_in_directory(string directory, vector<string> suffix) {
	vector<string> res;
	for(int i=0;i<suffix.size();++i) {
		vector<string> files = files_in_directory(directory, suffix[i]);
		res.insert(res.end(), files.begin(), files.end());
	}
	return res;
}

void compute_histogram(const vector< Mat > & img_lst, vector< Mat > & gradient_lst, const Size & size)
{
	const int hist_size[] = {256};
	const float range_0[]={0, 255};
	int channels[] = {0};
	const float* ranges[] = { range_0};

	vector< Mat >::const_iterator img = img_lst.begin();
	vector< Mat >::const_iterator end = img_lst.end();
	for (; img != end; ++img)
	{
		Mat hist;
		for(int i=0;i<3;++i){
			channels[0] = i;
			Mat m;
			calcHist(&*img, 1, channels, Mat(), m, 1, hist_size, ranges);
			hist.push_back(m);
		}
		gradient_lst.push_back(hist);
#ifdef _DEBUG
		imshow("gradient", get_hogdescriptor_visu(img->clone(), descriptors, size));
		waitKey(10);
#endif
	}
}

void compute_hog(const vector< Mat > & img_lst, vector< Mat > & gradient_lst, const Size & size)
{
	HOGDescriptor hog;
	hog.winSize = size;
//	hog.blockStride = Size(4,4);
	//hog.blockSize = Size(10,10);
	Mat gray;
	vector< Point > location;
	vector< float > descriptors;

	vector< Mat >::const_iterator img = img_lst.begin();
	vector< Mat >::const_iterator end = img_lst.end();
	for (; img != end; ++img)
	{
		cvtColor(*img, gray, COLOR_BGR2GRAY);
		hog.compute(gray, descriptors, Size(8, 8), Size(0, 0), location);
//		cout << descriptors.size() << endl;
		gradient_lst.push_back(Mat(descriptors).clone());
#ifdef _DEBUG
		imshow("gradient", get_hogdescriptor_visu(img->clone(), descriptors, size));
		waitKey(10);
#endif
	}
}

Ptr<SVM> train_hog_svm(const vector< Mat > & gradient_lst, const vector< int > & labels)
{
	/* Default values to train SVM */
	Ptr<SVM> svm = SVM::create();
	svm->setCoef0(0.0);
	svm->setDegree(1);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-3));
	svm->setGamma(0);
	svm->setKernel(SVM::LINEAR);
	svm->setNu(0.5);
	svm->setP(0.1); // for EPSILON_SVR, epsilon in loss function?
	svm->setC(1); // From paper, soft classifier
	svm->setType(SVM::EPS_SVR); // C_SVC; // EPSILON_SVR; // may be also NU_SVR; // do regression task

	Mat train_data;
	convert_to_ml(gradient_lst, train_data);

//	clog << "Start training...";
	svm->train(train_data, ROW_SAMPLE, Mat(labels));
//	clog << "...[done]" << endl;

	return svm;
}

Ptr<SVM> train_histogram_svm_on_dataset(vector<Mat>& pos_list, vector<Mat>& neg_list, Size image_size) {
	vector< Mat > gradient_lst;
	vector< int > labels;
	labels.assign(pos_list.size(), +1);
	labels.insert(labels.end(), neg_list.size(), -1);
	compute_histogram(pos_list, gradient_lst, image_size);
//	cout << pos_lst.size() << endl;
	compute_histogram(neg_list, gradient_lst, image_size);

	/* Default values to train SVM */
	Ptr<SVM> svm = SVM::create();
	svm->setCoef0(0.0);
	svm->setDegree(1);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1000, 1e-9));
//	svm->setGamma(0.5);
	svm->setGamma(0.25);
	svm->setKernel(SVM::LINEAR);
	svm->setNu(0.5);
	svm->setP(0.1); // for EPSILON_SVR, epsilon in loss function?
//	svm->setC(0.01); // From paper, soft classifier
	svm->setC(.25);
	svm->setType(SVM::EPS_SVR); // C_SVC; // EPSILON_SVR; // may be also NU_SVR; // do regression task

	Mat train_data;
//	cout << "moin" << endl;
	convert_to_ml(gradient_lst, train_data);

//	clog << "Start training...";
	svm->train(train_data, ROW_SAMPLE, Mat(labels));
//	cout << "Finished training." << endl;
	return svm;
}

Ptr<SVM> train_hog_svm_on_dataset(vector<Mat>& pos_list, vector<Mat>& neg_list, Size image_size) {
	vector< Mat > gradient_lst;
	vector< int > labels;
	labels.assign(pos_list.size(), +1);
	labels.insert(labels.end(), neg_list.size(), -1);
	compute_hog(pos_list, gradient_lst, image_size);
//	cout << pos_lst.size() << endl;
	compute_hog(neg_list, gradient_lst, image_size);

	/* Default values to train SVM */
	Ptr<SVM> svm = SVM::create();
	svm->setCoef0(0.0);
	svm->setDegree(1);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1000, 1e-9));
//	svm->setGamma(0.5);
	svm->setGamma(0.25);
	svm->setKernel(SVM::LINEAR);
	svm->setNu(0.5);
	svm->setP(0.1); // for EPSILON_SVR, epsilon in loss function?
//	svm->setC(0.01); // From paper, soft classifier
	svm->setC(.25);
	svm->setType(SVM::EPS_SVR); // C_SVC; // EPSILON_SVR; // may be also NU_SVR; // do regression task

	Mat train_data;
//	cout << "moin" << endl;
	convert_to_ml(gradient_lst, train_data);

//	clog << "Start training...";
	svm->train(train_data, ROW_SAMPLE, Mat(labels));
//	cout << "Finished training." << endl;
	return svm;
}

void train_hog_svm_and_store_in_file(string path_to_positive_samples, string path_to_negative_samples, string outFile) {
	vector< Mat > image_list_positive;
	vector< Mat > image_list_negative;
	vector< string > image_names_positive;
	vector< string > image_names_negative;
	const Size image_size(32, 32);
	load_images_with_names(path_to_positive_samples, image_list_positive, image_names_positive, image_size,0);
	load_images_with_names(path_to_negative_samples, image_list_negative, image_names_negative, image_size,0);
	Ptr<SVM> hog_svm = train_hog_svm_on_dataset(image_list_positive, image_list_negative, image_size);
	hog_svm->save(outFile);
}

void train_histogram_svm_and_store_in_file(string path_to_positive_samples, string path_to_negative_samples, string outFile) {
	vector< Mat > image_list_positive;
	vector< Mat > image_list_negative;
	vector< string > image_names_positive;
	vector< string > image_names_negative;
	const Size image_size(32, 32);
	load_images_with_names(path_to_positive_samples, image_list_positive, image_names_positive, image_size,0);
	load_images_with_names(path_to_negative_samples, image_list_negative, image_names_negative, image_size,0);
	Ptr<SVM> histogram_svm = train_histogram_svm_on_dataset(image_list_positive, image_list_negative, image_size);
	histogram_svm->save(outFile);
}


vector<string> get_keys(map<string, string> m) {
	vector<string> v;
	for(map<string,string>::iterator it = m.begin(); it != m.end(); ++it) {
	  v.push_back(it->first);
	}
	return v;
}

// From http://www.juergenwiki.de/work/wiki/doku.php?id=public:hog_descriptor_computation_and_visualization
Mat get_hogdescriptor_visu(const Mat& color_origImg, vector<float>& descriptorValues, const Size & size)
{
	const int DIMX = size.width;
	const int DIMY = size.height;
	float zoomFac = 3;
	Mat visu;
	resize(color_origImg, visu, Size((int)(color_origImg.cols*zoomFac), (int)(color_origImg.rows*zoomFac)));

	int cellSize = 8;
	int gradientBinSize = 9;
	float radRangeForOneBin = (float)(CV_PI / (float)gradientBinSize); // dividing 180° into 9 bins, how large (in rad) is one bin?

																	   // prepare data structure: 9 orientation / gradient strenghts for each cell
	int cells_in_x_dir = DIMX / cellSize;
	int cells_in_y_dir = DIMY / cellSize;
	float*** gradientStrengths = new float**[cells_in_y_dir];
	int** cellUpdateCounter = new int*[cells_in_y_dir];
	for (int y = 0; y<cells_in_y_dir; y++)
	{
		gradientStrengths[y] = new float*[cells_in_x_dir];
		cellUpdateCounter[y] = new int[cells_in_x_dir];
		for (int x = 0; x<cells_in_x_dir; x++)
		{
			gradientStrengths[y][x] = new float[gradientBinSize];
			cellUpdateCounter[y][x] = 0;

			for (int bin = 0; bin<gradientBinSize; bin++)
				gradientStrengths[y][x][bin] = 0.0;
		}
	}

	// nr of blocks = nr of cells - 1
	// since there is a new block on each cell (overlapping blocks!) but the last one
	int blocks_in_x_dir = cells_in_x_dir - 1;
	int blocks_in_y_dir = cells_in_y_dir - 1;

	// compute gradient strengths per cell
	int descriptorDataIdx = 0;
	int cellx = 0;
	int celly = 0;

	for (int blockx = 0; blockx<blocks_in_x_dir; blockx++)
	{
		for (int blocky = 0; blocky<blocks_in_y_dir; blocky++)
		{
			// 4 cells per block ...
			for (int cellNr = 0; cellNr<4; cellNr++)
			{
				// compute corresponding cell nr
				cellx = blockx;
				celly = blocky;
				if (cellNr == 1) celly++;
				if (cellNr == 2) cellx++;
				if (cellNr == 3)
				{
					cellx++;
					celly++;
				}

				for (int bin = 0; bin<gradientBinSize; bin++)
				{
					float gradientStrength = descriptorValues[descriptorDataIdx];
					descriptorDataIdx++;

					gradientStrengths[celly][cellx][bin] += gradientStrength;

				} // for (all bins)


				  // note: overlapping blocks lead to multiple updates of this sum!
				  // we therefore keep track how often a cell was updated,
				  // to compute average gradient strengths
				cellUpdateCounter[celly][cellx]++;

			} // for (all cells)


		} // for (all block x pos)
	} // for (all block y pos)


	  // compute average gradient strengths
	for (celly = 0; celly<cells_in_y_dir; celly++)
	{
		for (cellx = 0; cellx<cells_in_x_dir; cellx++)
		{

			float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];

			// compute average gradient strenghts for each gradient bin direction
			for (int bin = 0; bin<gradientBinSize; bin++)
			{
				gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
			}
		}
	}

	// draw cells
	for (celly = 0; celly<cells_in_y_dir; celly++)
	{
		for (cellx = 0; cellx<cells_in_x_dir; cellx++)
		{
			int drawX = cellx * cellSize;
			int drawY = celly * cellSize;

			int mx = drawX + cellSize / 2;
			int my = drawY + cellSize / 2;

			rectangle(visu, Point((int)(drawX*zoomFac), (int)(drawY*zoomFac)), Point((int)((drawX + cellSize)*zoomFac), (int)((drawY + cellSize)*zoomFac)), Scalar(100, 100, 100), 1);

			// draw in each cell all 9 gradient strengths
			for (int bin = 0; bin<gradientBinSize; bin++)
			{
				float currentGradStrength = gradientStrengths[celly][cellx][bin];

				// no line to draw?
				if (currentGradStrength == 0)
					continue;

				float currRad = bin * radRangeForOneBin + radRangeForOneBin / 2;

				float dirVecX = cos(currRad);
				float dirVecY = sin(currRad);
				float maxVecLen = (float)(cellSize / 2.f);
				float scale = 2.5; // just a visualization scale, to see the lines better

								   // compute line coordinates
				float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
				float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
				float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
				float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;

				// draw gradient visualization
				line(visu, Point((int)(x1*zoomFac), (int)(y1*zoomFac)), Point((int)(x2*zoomFac), (int)(y2*zoomFac)), Scalar(0, 255, 0), 1);

			} // for (all bins)

		} // for (cellx)
	} // for (celly)


	  // don't forget to free memory allocated by helper data structures!
	for (int y = 0; y<cells_in_y_dir; y++)
	{
		for (int x = 0; x<cells_in_x_dir; x++)
		{
			delete[] gradientStrengths[y][x];
		}
		delete[] gradientStrengths[y];
		delete[] cellUpdateCounter[y];
	}
	delete[] gradientStrengths;
	delete[] cellUpdateCounter;

	return visu;
} // get_hogdescriptor_visu

void get_difference_vector(vector<float> v1, vector<float> v2, vector<float>& res) {
	int n = min(v1.size(), v2.size());
	for(int i=0;i<n;++i) {
		res.push_back(v1[i] - v2[i]);
	}
}

double get_difference_sum(vector<float> v1, vector<float> v2) {
	double res = 0;
	int n = min(v1.size(), v2.size());
	for(int i=0;i<n;++i) {
		res += abs(v1[i]-v2[i]);
	}
	return res;
}

void get_mean_var_vector(Mat m, vector<float>& means, vector<float>& vars) {
	int no_rows = m.rows;
	for(int i=0;i<no_rows;++i) {
		Mat meanMat, stddevMat;
		meanStdDev(m.row(i), meanMat, stddevMat);
		means.push_back(meanMat.at<double>(0,0));
		vars.push_back(stddevMat.at<double>(0,0));
	}
}

void get_mean_var_vector(vector< Mat >& m, vector<float>& means, vector<float>& vars) {
	int no_rows = m.size();
	Mat mmat(Size(m.size(), m[0].rows), CV_32F);
	for(int i=0;i<no_rows;++i) {
		for(int j=0;j<m[0].rows;++j) {
			mmat.at<double>(i,j) = m[i].at<float>(j,0);
		}
	}
	get_mean_var_vector(mmat, means, vars);
}

void get_mean_var_vector(vector< vector<float> >& m, vector<float>& means, vector<float>& vars) {
	const int no_rows = m[0].size();
	const int no_cols = m.size();
	Mat mmat(Size(no_rows, no_cols), CV_32F);
	for(int r=0;r<no_rows;++r) {
		for(int c=0;c<no_cols;++c) {
			mmat.at<double>(r,c)=m[c][r];
		}
	}
	get_mean_var_vector(mmat, means, vars);
}

