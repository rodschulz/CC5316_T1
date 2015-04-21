/**
 * Author: rodrigo
 * 2015
 */
#include "Printer.h"

#define OUTPUT_FOLDER		"./output/"
#define WIDTH			1280
#define HEIGHT			960
#define HORIZONTAL_OFFSET	(WIDTH / 2)
#define VERTICAL_OFFSET		(HEIGHT / 2)

// Color used to draw triangles
static Scalar LIGHT_GRAY = Scalar(190, 190, 190);
static Scalar BLACK = Scalar(0, 0, 0);

Printer::Printer()
{
	conversionRate = 4;
	step = 5;
}

Printer::~Printer()
{
}

void Printer::saveImage(const string &_outputName, const Mat &_image)
{
	imwrite(OUTPUT_FOLDER + _outputName, _image);
}

Point Printer::convert(const double _x, const double _y)
{
	Printer* printer = getInstance();
	int x = printer->toPixel(_x) + HORIZONTAL_OFFSET;
	int y = VERTICAL_OFFSET - printer->toPixel(_y);
	return Point(x, y);
}

void Printer::drawPoint(Mat &_image, const float _x, const float _y, const Scalar &_color)
{
	Point p = convert(_x, _y);
	circle(_image, p, 1, _color, -2);
}

Mat Printer::generateBaseImage()
{
	Printer* printer = getInstance();
	Mat image(HEIGHT, WIDTH, CV_8UC3, LIGHT_GRAY);

	line(image, Point(0, VERTICAL_OFFSET), Point(WIDTH, VERTICAL_OFFSET), BLACK);
	line(image, Point(HORIZONTAL_OFFSET, 0), Point(HORIZONTAL_OFFSET, HEIGHT), BLACK);

	int step = getInstance()->step;
	int tickSize = printer->toPixel(0.3);
	int position = step;
	int tickOffsetH = 5;
	int tickOffsetV = 10;

	// X ticks
	int pos = printer->toPixel(position);
	while (pos <= WIDTH)
	{
		line(image, Point(pos + HORIZONTAL_OFFSET, VERTICAL_OFFSET - tickSize), Point(pos + HORIZONTAL_OFFSET, VERTICAL_OFFSET + tickSize), BLACK);
		putText(image, to_string(position), Point(pos + HORIZONTAL_OFFSET - tickOffsetH, VERTICAL_OFFSET + tickSize + tickOffsetV), CV_FONT_HERSHEY_SIMPLEX, 0.3, BLACK);

		line(image, Point(HORIZONTAL_OFFSET - pos, VERTICAL_OFFSET - tickSize), Point(HORIZONTAL_OFFSET - pos, VERTICAL_OFFSET + tickSize), BLACK);
		putText(image, to_string(-position), Point(HORIZONTAL_OFFSET - pos - tickOffsetH - 5, VERTICAL_OFFSET + tickSize + tickOffsetV), CV_FONT_HERSHEY_SIMPLEX, 0.3, BLACK);

		position += step;
		pos = printer->toPixel(position);
	}

	position = step;
	pos = printer->toPixel(position);
	tickOffsetH = 15;
	tickOffsetV = 3;

	// Y ticks
	while (pos <= HEIGHT)
	{
		line(image, Point(HORIZONTAL_OFFSET - tickSize, VERTICAL_OFFSET - pos), Point(HORIZONTAL_OFFSET + tickSize, VERTICAL_OFFSET - pos), BLACK);
		putText(image, to_string(position), Point(HORIZONTAL_OFFSET - tickSize - tickOffsetH, VERTICAL_OFFSET - pos + tickOffsetV), CV_FONT_HERSHEY_SIMPLEX, 0.3, BLACK);

		line(image, Point(HORIZONTAL_OFFSET - tickSize, VERTICAL_OFFSET + pos), Point(HORIZONTAL_OFFSET + tickSize, VERTICAL_OFFSET + pos), BLACK);
		putText(image, to_string(-position), Point(HORIZONTAL_OFFSET - tickSize - tickOffsetH - 10, VERTICAL_OFFSET + pos + tickOffsetV), CV_FONT_HERSHEY_SIMPLEX, 0.3, BLACK);

		position += step;
		pos = printer->toPixel(position);
	}

	return image;
}

Printer *Printer::getInstance()
{
	static Printer *instance = new Printer();
	return instance;
}

void Printer::calculateConversionRate(const double _width, const double _height)
{
	double horizontalRate = WIDTH / _width;
	double verticalRate = HEIGHT / _height;

	conversionRate = horizontalRate < verticalRate ? horizontalRate : verticalRate;

	double size = _width > _height ? _width : _height;
	size /= 10;

	double distance = numeric_limits<double>::max();
	double last = step;
	bool state = true;
	while (true)
	{
		double localDist = abs(step - size);
		if (localDist >= distance)
		{
			step = last;
			break;
		}

		distance = localDist;
		last = step;

		if (state)
			step *= 2;
		else
			step *= 5;

		state = !state;
	}
}

int Printer::toPixel(const double _value) const
{
	return (int) (_value * conversionRate);
}

void Printer::printTrajectory(vector<Mat> &_poses, const string &_name, const Scalar &_color)
{
	Mat image = generateBaseImage();
	for (Mat m : _poses)
		drawPoint(image, m.at<double>(0, 3), m.at<double>(2, 3), _color);
	saveImage(_name, image);
}
