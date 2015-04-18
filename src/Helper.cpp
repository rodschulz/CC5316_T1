/**
 * Author: rodrigo
 * 2015
 */
#include "Helper.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <opencv2/highgui/highgui.hpp>

Helper::Helper()
{
}

Helper::~Helper()
{
}

vector<DMatch> Helper::getMatches(const Mat &_descriptor1, const Mat &_descriptor2, const DescriptorMatcher &_matcher)
{
	vector<DMatch> matches;
	_matcher.match(_descriptor1, _descriptor2, matches);

	// Calculation of max and min distances between keypoints
	double max_dist = 0;
	double min_dist = 100;
	for (int i = 0; i < _descriptor1.rows; i++)
	{
		double dist = matches[i].distance;
		min_dist = dist < min_dist ? dist : min_dist;
		max_dist = dist > max_dist ? dist : max_dist;
	}

	// Extract good matches
	vector<DMatch> goodMatches;
	for (int i = 0; i < _descriptor1.rows; i++)
	{
		if (matches[i].distance <= max(2 * min_dist, 0.02))
			goodMatches.push_back(matches[i]);
	}

	return goodMatches;
}

void Helper::showMatches(const Mat &_image1, const Mat &_image2, const vector<KeyPoint> &_keypoints1, const vector<KeyPoint> &_keypoints2, const vector<DMatch> &_matches)
{
	Mat imgMatches;
	drawMatches(_image1, _keypoints1, _image2, _keypoints2, _matches, imgMatches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	// Show detected matches
	namedWindow("Good Matches", CV_WINDOW_AUTOSIZE);
	imshow("Good Matches", imgMatches);
	waitKey(0);
}

void Helper::setMatrixRow(Mat &_A, const int _rowIndex, const KeyPoint &_keypoint1, const KeyPoint &_keypoint2)
{
	float x1 = _keypoint1.pt.x;
	float y1 = _keypoint1.pt.y;
	float x2 = _keypoint2.pt.x;
	float y2 = _keypoint2.pt.y;

	_A.at<float>(_rowIndex, 0) = x1 * x2;
	_A.at<float>(_rowIndex, 1) = y1 * x2;
	_A.at<float>(_rowIndex, 2) = x2;
	_A.at<float>(_rowIndex, 3) = x1 * y2;
	_A.at<float>(_rowIndex, 4) = y1 * y2;
	_A.at<float>(_rowIndex, 5) = y2;
	_A.at<float>(_rowIndex, 6) = x1;
	_A.at<float>(_rowIndex, 7) = y1;
	_A.at<float>(_rowIndex, 8) = 1;
}

void Helper::loadInput(vector<Mat> &_destination, const string &_inputFile)
{
	_destination.clear();

	string line;
	ifstream file;
	file.open(_inputFile.c_str(), fstream::in);
	if (file.is_open())
	{
		while (getline(file, line))
		{
			if (line.empty() || line.at(0) == '#')
				continue;
			_destination.push_back(imread(line, CV_LOAD_IMAGE_GRAYSCALE));
		}
		file.close();
	}
	else
		cout << "Unable to open input: " << _inputFile << "\n";
}
