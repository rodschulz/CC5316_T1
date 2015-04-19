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
	// Generates a base image
	static Mat generateBaseImage();
	// Saves the given image to disk
	static void saveImage(const string &_outputName, const Mat &_image);
	// Returns an instance of the current singleton printer class
	static Printer *getInstance();
	// Calculates the conversion rate according to the given dimensions
	void calculateConversionRate(const double _width, const double _height);
	// Draws the given point in the given image
	static void drawPoint(Mat &_image, const float _x, const float _y, const Scalar &_color);

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
