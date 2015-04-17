/**
 * Author: rodrigo
 * 2015
 */
#pragma once

#include "stdio.h"
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace cv;

class Helper
{
public:
	// Prints the given matrix in the stdout
	template<class T> static void printMatrix(const Mat &_matrix)
	{
		for (int i = 0; i < _matrix.rows; i++)
		{
			for (int j = 0; j < _matrix.cols; j++)
			{
				printf("%5f\t", _matrix.at<T>(i, j));
			}
			printf("\n");
		}
	}
	// Returns the matching points between both descriptors
	static vector<DMatch> getMatches(const Mat &_descriptor1, const Mat &_descriptor2, const DescriptorMatcher &_matcher);
	// Shows an image displaying the matches between images
	static void showMatches(const Mat &_image1, const Mat &_image2, const vector<KeyPoint> &_keypoints1, const vector<KeyPoint> &_keypoints2, const vector<DMatch> &_matches);

private:
	Helper();
	~Helper();
};
