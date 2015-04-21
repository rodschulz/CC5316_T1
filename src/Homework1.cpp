#include <stdlib.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "Helper.h"
#include "Printer.h"
#include "Loader.h"

using namespace std;
using namespace cv;

typedef enum DebugLevel
{
	NONE, LOW, MEDIUM, HIGH
} DebugLevel;

bool test(const Point2f &pt1, const Point2f &pt2, const Mat &R1, const Mat &t)
{
	Mat num = (R1.row(0) - pt2.x * R1.row(2)) * t;
	Mat aux(3, 1, CV_64FC1);
	aux.at<double>(0, 0) = pt2.x;
	aux.at<double>(1, 0) = pt2.y;
	aux.at<double>(2, 0) = 1;
	Mat denom = (R1.row(0) - pt2.x * R1.row(2)) * aux;

	double z1 = num.at<double>(0, 0) / denom.at<double>(0, 0);

	Mat x1(3, 1, CV_64FC1);
	x1.at<double>(0, 0) = pt1.x * z1;
	x1.at<double>(1, 0) = pt1.y * z1;
	x1.at<double>(2, 0) = z1;

	Mat x2 = R1.t() * (x1 - t);

	cout << "z1: " << x1.at<double>(2, 0) << " z2: " << x2.at<double>(2, 0) << "\n";

	return x1.at<double>(2, 0) >= 0 && x2.at<double>(2, 0) >= 0;
}

int main(int _nargs, char **_vargs)
{
	DebugLevel debugLevel = LOW;
	bool showImages = false;

	system("rm -rf ./output/*");

	if (_nargs < 2)
	{
		cout << "Not enough arguments\nUsage:\n\tHomework1 <input_file>\n\n";
		return EXIT_FAILURE;
	}

	// Get the calibration matrix
	Mat K = Mat::zeros(3, 3, CV_64FC1);
	Loader::loadCalibrationMatrix(K, _vargs[1]);
	Mat Kt = Mat::zeros(3, 3, CV_64FC1);
	transpose(K, Kt);

	cout << "K:\n";
	Printer::printMatrix<double>(K, 3);

	// Load images
	vector<Mat> images;
	Loader::loadInput(images, _vargs[1]);

	// Get ground truth
	vector<Mat> groundTruth;
	Loader::loadGroundTruth(groundTruth, _vargs[1], images.size());

	// Print groundtruth trajectory
	Printer::getInstance()->calculateConversionRate(400, 400);
	Printer::printTrajectory(groundTruth, "groundTruth.png", RED);

	Ptr<FeatureDetector> featureExtractor = FeatureDetector::create("HARRIS");
	Ptr<DescriptorExtractor> descriptorExtractor = DescriptorExtractor::create("SIFT");
	FlannBasedMatcher matcher;

	vector<vector<KeyPoint>> keypoints;
	vector<Mat> descriptors;
	vector<Mat> trajectory;
	trajectory.push_back(Mat(3, 4, CV_64FC1));

	// Process data
	initModule_nonfree();
	for (size_t k = 0; k < images.size(); k++)
	{
		Mat image = images[k];

		cout << "Processing image " << k << "\n";
		// Keypoints extraction
		keypoints.push_back(vector<KeyPoint>());
		featureExtractor->detect(image, keypoints.back());
		// Feature extraction
		descriptors.push_back(Mat());
		descriptorExtractor->compute(image, keypoints.back(), descriptors.back());

		if (k > 0)
		{
			// Match points between images
			vector<DMatch> matches = Helper::getMatches(descriptors[k - 1], descriptors[k], matcher);

			if (showImages)
				Helper::showMatches(images[k - 1], images[k], keypoints[k - 1], keypoints[k], matches);

			if (matches.size() >= 8)
			{
//				Mat A(8, 9, CV_64FC1);
//				for (int i = 0; i < 8; i++)
//					Helper::setMatrixRow(A, i, keypoints[k - 1][matches[i].queryIdx], keypoints[k][matches[i].trainIdx]);
//
//				// Perform SVD over A
//				SVD decompA(A, SVD::FULL_UV);
//				Mat F(3, 3, CV_64FC1);
//				F.at<double>(0, 0) = decompA.vt.at<double>(8, 0);
//				F.at<double>(0, 1) = decompA.vt.at<double>(8, 1);
//				F.at<double>(0, 2) = decompA.vt.at<double>(8, 2);
//				F.at<double>(1, 0) = decompA.vt.at<double>(8, 3);
//				F.at<double>(1, 1) = decompA.vt.at<double>(8, 4);
//				F.at<double>(1, 2) = decompA.vt.at<double>(8, 5);
//				F.at<double>(2, 0) = decompA.vt.at<double>(8, 6);
//				F.at<double>(2, 1) = decompA.vt.at<double>(8, 7);
//				F.at<double>(2, 2) = decompA.vt.at<double>(8, 8);

				vector<Point2f> points1, points2;
				for (DMatch m : matches)
				{
					points1.push_back(keypoints[k - 1][m.queryIdx].pt);
					points2.push_back(keypoints[k][m.trainIdx].pt);
				}

				Mat F = findFundamentalMat(points1, points2, FM_RANSAC, 3, 0.99);

				if (debugLevel >= MEDIUM)
					Printer::printMatrix<double>(F, 5, "F:");

				Mat E = Kt * F * K;
				if (debugLevel >= MEDIUM)
					Printer::printMatrix<double>(E, 5, "E:");

				SVD decompE(E, SVD::FULL_UV);
				if (debugLevel >= HIGH)
				{
					Printer::printMatrix<double>(decompE.u, 5, "Decomp.U:");
					Printer::printMatrix<double>(decompE.w, 5, "Decomp.Sigma:");
					Printer::printMatrix<double>(decompE.vt, 5, "Decomp.Vt:");
				}

				Mat W = Mat::zeros(3, 3, CV_64FC1);
				W.at<double>(0, 1) = -1;
				W.at<double>(1, 0) = 1;
				W.at<double>(2, 2) = 1;
				Mat Z = Mat::zeros(3, 3, CV_64FC1);
				Z.at<double>(0, 1) = 1;
				Z.at<double>(1, 0) = -1;

//				Mat sigma = Mat::zeros(3, 3, CV_64FC1);
//				sigma.at<double>(0, 0) = decompE.w.at<double>(0, 0);
//				sigma.at<double>(1, 1) = decompE.w.at<double>(0, 1);
//				sigma.at<double>(2, 2) = decompE.w.at<double>(0, 2);
//				Helper::printMatrix<double>(sigma, 5, "sigma:");
//				Mat t = decompE.vt.t() * W * sigma * decompE.vt;
//				Helper::printMatrix<double>(t, 5, "t:");

//				Mat tx = decompE.vt.t() * Z * decompE.vt;
//				Helper::printMatrix<double>(tx, 5, "tx:");

				Mat R1 = decompE.u * W.t() * decompE.vt;
				Mat R2 = decompE.u * W * decompE.vt;
//				Helper::printMatrix<double>(R1.t() * R1, 5, "R1");
//				cout << "det1 " << determinant(R1) << "\n";
//				Helper::printMatrix<double>(R2.t() * R2, 5, "R2");
//				cout << "det2 " << determinant(R2) << "\n";
//
//				Mat S1 = -decompE.u * Z * decompE.u.t();
//				Mat S2 = decompE.u * Z * decompE.u.t();
//				Helper::printMatrix<double>(S1 * R1, 5, "S1*R1");
//				Helper::printMatrix<double>(S2 * R2, 5, "S2*R2");

				if (debugLevel >= HIGH)
				{
					Printer::printMatrix<double>(decompE.u.col(2), 5, "U2:");
					Printer::printMatrix<double>(R1, 5, "R1");
					Printer::printMatrix<double>(R2, 5, "R2");
				}

				Mat x(4, 1, CV_64FC1);
				x.at<double>(0, 0) = 5;
				x.at<double>(1, 0) = 5;
				x.at<double>(2, 0) = 5;
				x.at<double>(3, 0) = 1;

//				Mat P1 = Mat::zeros(3, 4, CV_64FC1);
//				P1.at<double>(0, 0) = 1;
//				P1.at<double>(1, 1) = 1;
//				P1.at<double>(2, 2) = 1;
				//Helper::printMatrix<double>(P1, 5, "P1");
				//Helper::printMatrix<double>(P1 * x, 5, "P1 * x");

//				Mat P2a;
//				hconcat(R1, decompE.u.col(2), P2a);
////				Helper::printMatrix<double>(P2a, 5, "P2a");
//				Helper::printMatrix<double>(P2a * x, 5, "P2a * x");
//
//				Mat P2b;
//				hconcat(R1, -decompE.u.col(2), P2b);
////				Helper::printMatrix<double>(P2b, 5, "P2b");
//				Helper::printMatrix<double>(P2b * x, 5, "P2b * x");
//
//				Mat P2c;
//				hconcat(R2, decompE.u.col(2), P2c);
////				Helper::printMatrix<double>(P2c, 5, "P2c");
//				Helper::printMatrix<double>(P2c * x, 5, "P2c * x");
//
//				Mat P2d;
//				hconcat(R2, -decompE.u.col(2), P2d);
////				Helper::printMatrix<double>(P2d, 5, "P2d");
//				Helper::printMatrix<double>(P2d * x, 5, "P2d * x");

				Mat delta = Mat::zeros(3, 1, CV_64FC1);

				Point2f pt1 = keypoints[k - 1][matches[0].queryIdx].pt;
				Point2f pt2 = keypoints[k - 1][matches[0].trainIdx].pt;
				if (test(pt1, pt2, R1, decompE.u.col(2)))
					delta = R1 * decompE.u.col(2);

				if (test(pt1, pt2, R1, -decompE.u.col(2)))
					delta = R1 * (-decompE.u.col(2));

				if (test(pt1, pt2, R2, decompE.u.col(2)))
					delta = R2 * decompE.u.col(2);

				if (test(pt1, pt2, R2, -decompE.u.col(2)))
					delta = R2 * (-decompE.u.col(2));

				Mat pose = trajectory.back().clone();
//				pose.col(3) += decompE.u.col(2);
				pose.col(3) += delta;
				trajectory.push_back(pose);

				if (debugLevel >= LOW)
					Printer::printMatrix<double>(trajectory.back(), 5, "Pose:");
			}
		}
	}

	cout << "Printing final trajectory\n";
	Printer::printTrajectory(trajectory, "trajectory.png", GREEN);

	return EXIT_SUCCESS;
}
