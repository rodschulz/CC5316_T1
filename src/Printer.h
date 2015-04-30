/**
 * Author: rodrigo
 * 2015
 */
#pragma once

#include <vector>
#include <string>
#include "opencv/cxcore.h"
#include "opencv/cv.h"
#include "opencv/highgui.h"

using namespace std;
using namespace cv;

class Printer
{
public:
	// Prints the given matrix in the stdout
	template<class T> static void printMatrix(const Mat &_matrix, const int _precision = 1, const string &_name = "")
	{
		string format = "%- 15." + to_string(_precision) + "f\t";

		printf("%s\n", _name.c_str());
		for (int i = 0; i < _matrix.rows; i++)
		{
			for (int j = 0; j < _matrix.cols; j++)
			{
				printf(format.c_str(), _matrix.at<T>(i, j));
			}
			printf("\n");
		}
	}

	// Generates a base image
	static Mat generateBaseImage();
	// Saves the given image to disk
	static void saveImage(const string &_outputName, const Mat &_image);
	// Returns an instance of the current singleton printer class
	static Printer *getInstance();
	// Draws the given point in the given image
	static void drawPoint(Mat &_image, const float _x, const float _y, const Scalar &_color);
	// prints the given trajectory
	static void printTrajectory(vector<Mat> &_poses, const string &_name, const Scalar &_color);
	// prints the given trajectory
	static void printTrajectory2(vector<Mat> &_poses, const string &_name, const Scalar &_color);

	// Calculates the conversion rate according to the given dimensions
	void calculateConversionRate(const double _width, const double _height);

private:
	Printer();
	~Printer();

	double conversionRate;
	int step;

	// Converts the given value to its pixel equivalent
	int toPixel(const double _value) const;
	// Converts a point from xy coordinates to pixels
	static Point convert(const double _x, const double _y);
};
