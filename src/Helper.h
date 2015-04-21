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

#define BLUE Scalar(255, 0, 0)
#define GREEN Scalar(0, 255, 0)
#define RED Scalar(0, 0, 255)
#define LIGHT_BLUE Scalar(0, 255, 255)

class Helper
{
public:
	// Returns the matching points between both descriptors
	static vector<DMatch> getMatches(const Mat &_descriptor1, const Mat &_descriptor2, const DescriptorMatcher &_matcher);
	// Shows an image displaying the matches between images
	static void showMatches(const Mat &_image1, const Mat &_image2, const vector<KeyPoint> &_keypoints1, const vector<KeyPoint> &_keypoints2, const vector<DMatch> &_matches);

	// Sets the values of a row in a matrix to perform a SVD, in order to calculate the values of the fundamental matrix
	static void setMatrixRow(Mat &_A, const int _rowIndex, const KeyPoint &_keypoint1, const KeyPoint &_keypoint2);

private:
	Helper();
	~Helper();
};
