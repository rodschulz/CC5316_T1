/**
 * Author: rodrigo
 * 2015
 */
#pragma once

#include <string>
#include "stdio.h"
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>

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
	static vector<DMatch> getMatches(const Mat &_descriptor1, const Mat &_descriptor2, const FlannBasedMatcher &_matcher);
	// Shows an image displaying the matches between images
	static void showMatches(const Mat &_image1, const Mat &_image2, const vector<KeyPoint> &_keypoints1, const vector<KeyPoint> &_keypoints2, const vector<DMatch> &_matches);
	// Calculates the fundamental matrix from the set of given keypoints and matches
	static void getFundamentalMatrix(Mat &_F, const vector<DMatch> &_matches, const vector<KeyPoint> &_keypointsQuery, const vector<KeyPoint> &_keypointsTrain);
	// Calculates the essential matrix from the fundamental and calibration matrices
	static void getEssentialMatrix(Mat &_E, const Mat &_K, const Mat &_F);
	// Calculates the motion between the two images
	static void calculateMotion(const Mat &_E, const Mat &_K, Mat &_R, Mat &_t, Mat &_points3D, const vector<KeyPoint> &_queryKeypoints, const vector<KeyPoint> &_trainKeypoints, const vector<DMatch> &_matches);
	// Triangulates a set of matches to get the estimation of corresponding 3d points
	static int triangulatePoints(const vector<KeyPoint> &_queryKeypoints, const vector<KeyPoint> &_trainKeypoints, const Mat &_K, const Mat &_R, const Mat &_t, Mat &_points3D);
	// Return the points associated to each keypoint
	static void extractPoints(const vector<KeyPoint> &_trainKeypoints, const vector<KeyPoint> &_queryKeypoints, const vector<DMatch> &_matches, vector<Point2f> &_trainPoints, vector<Point2f> &_queryPoints);

private:
	Helper();
	~Helper();

	// Sets the values of a row in a matrix to perform a SVD, in order to calculate the values of the fundamental matrix
	static void setMatrixRow(Mat &_A, const int _rowIndex, const KeyPoint &_keypoint1, const KeyPoint &_keypoint2);
	// Comparison method used to compare to matches according to their distances
	static bool compareMatches(const DMatch &_match1, const DMatch &_match2);
};
