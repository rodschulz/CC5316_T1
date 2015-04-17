/**
 * Author: rodrigo
 * 2015
 */
#pragma once

#include "stdio.h"
#include <opencv2/core/core.hpp>

using namespace cv;

class Helper
{
public:
	template<class T> static void printMatrix(const Mat &_matrix)
	{
		for (int i = 0; i < _matrix.rows; i++)
		{
			for (int j = 0; j < _matrix.cols; j++)
			{
				printf("%5f\t", _matrix.at < T > (i, j));
			}
			printf("\n");
		}
	}

private:
	Helper();
	~Helper();
};
