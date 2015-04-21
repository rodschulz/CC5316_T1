/**
 * Author: rodrigo
 * 2015
 */
#pragma once

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace std;
using namespace cv;

class Loader
{
public:
	// Loads a set of images
	static void loadInput(vector<Mat> &_destination, const string &_inputFile);
	// Loads the calibration matrix from file
	static void loadCalibrationMatrix(Mat &_K, const string &_inputFile);
	// Loads the groundtruth data from the given input file
	static void loadGroundTruth(vector<Mat> &_poses, const string &_inputFile, const int _length);
private:
	Loader();
	~Loader();
};
