/**
 * Author: rodrigo
 * 2015
 */
#pragma once

#include <string>
#include "stdio.h"
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace cv;
using namespace std;

class Helper
{
public:
	// Prints the given matrix in the stdout
	template<class T> static void printMatrix(const Mat &_matrix, const int _precision = 1)
	{
		string format = "%- 15." + to_string(_precision) + "f\t";

		for (int i = 0; i < _matrix.rows; i++)
		{
			for (int j = 0; j < _matrix.cols; j++)
			{
				printf(format.c_str(), _matrix.at<T>(i, j));
			}
			printf("\n");
		}
	}
	// Returns the matching points between both descriptors
	static vector<DMatch> getMatches(const Mat &_descriptor1, const Mat &_descriptor2, const DescriptorMatcher &_matcher);
	// Shows an image displaying the matches between images
	static void showMatches(const Mat &_image1, const Mat &_image2, const vector<KeyPoint> &_keypoints1, const vector<KeyPoint> &_keypoints2, const vector<DMatch> &_matches);
	// Loads a set of images
	static void loadInput(vector<Mat> &_destination, const string &_inputFile);
	// Loads the calibration matrix from file
	static void loadCalibrationMatrix(Mat &_K, const string &_inputFile);

	// Sets the values of a row in a matrix to perform a SVD, in order to calculate the values of the fundamental matrix
	static void setMatrixRow(Mat &_A, const int _rowIndex, const KeyPoint &_keypoint1, const KeyPoint &_keypoint2);

private:
	Helper();
	~Helper();
};
