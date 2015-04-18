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
	int lineCount = 0;

	string line;
	ifstream file;
	file.open(_inputFile.c_str(), fstream::in);
	if (file.is_open())
	{
		while (getline(file, line))
		{
			if (line.empty() || line.at(0) == '#')
				continue;

			lineCount++;
			if (lineCount <= 2)
				continue;

			_destination.push_back(imread(line, CV_LOAD_IMAGE_GRAYSCALE));
		}
		file.close();
	}
	else
		cout << "Unable to open input: " << _inputFile << "\n";
}

void Helper::loadCalibrationMatrix(Mat &_K, const string &_inputFile)
{
	bool firstLine = true;
	string format = "";

	string line;
	ifstream file;
	file.open(_inputFile.c_str(), fstream::in);
	if (file.is_open())
	{
		while (getline(file, line))
		{
			if (line.empty() || line.at(0) == '#')
				continue;

			if (firstLine)
			{
				format = line;
				firstLine = false;
			}
			else
			{
				if (format.compare("FORMAT_KITTI") == 0)
				{
					string matrixLine;
					ifstream matrixFile;
					matrixFile.open(line.c_str(), fstream::in);

					if (matrixFile.is_open())
					{
						getline(matrixFile, matrixLine);

						// Parse string line
						vector<string> tokens;
						istringstream iss(matrixLine);
						copy(istream_iterator<string>(iss), istream_iterator<string>(), back_inserter(tokens));

						_K.at<float>(0, 0) = stof(tokens[1]);
						_K.at<float>(0, 1) = stof(tokens[2]);
						_K.at<float>(0, 2) = stof(tokens[3]);
						_K.at<float>(0, 3) = stof(tokens[4]);

						_K.at<float>(1, 0) = stof(tokens[5]);
						_K.at<float>(1, 1) = stof(tokens[6]);
						_K.at<float>(1, 2) = stof(tokens[7]);
						_K.at<float>(1, 3) = stof(tokens[8]);

						_K.at<float>(2, 0) = stof(tokens[9]);
						_K.at<float>(2, 1) = stof(tokens[10]);
						_K.at<float>(2, 2) = stof(tokens[11]);
						_K.at<float>(2, 3) = stof(tokens[12]);
					}
					else
						cout << "ERROR: Can't read calibration matrix\n";

					matrixFile.close();
					break;
				}
				else if (format.compare("FORMAT_COLLECTION") == 0)
				{
					string matrixLine;
					ifstream matrixFile;
					matrixFile.open(line.c_str(), fstream::in);

					if (matrixFile.is_open())
					{
						int lineCount = 1;
						while (getline(matrixFile, matrixLine))
						{
							switch (lineCount)
							{
								case 12:
								case 15:
								case 18:
								case 21:
								{
									int begin = matrixLine.find_first_of('>');
									int end = matrixLine.find_first_of('<', begin);
									string number = matrixLine.substr(begin + 1, end - begin - 1);

									switch (lineCount)
									{
										case 12:
											_K.at<float>(0, 0) = stof(number);
										break;
										case 15:
											_K.at<float>(1, 1) = stof(number);
										break;
										case 18:
											_K.at<float>(0, 2) = stof(number);
										break;
										case 21:
											_K.at<float>(1, 2) = stof(number);
										break;
									}
								}
								break;
							}

							lineCount++;
						}
					}
					matrixFile.close();
					break;
				}
			}
		}
		file.close();
	}
	else
		cout << "Unable to open input: " << _inputFile << "\n";
}
