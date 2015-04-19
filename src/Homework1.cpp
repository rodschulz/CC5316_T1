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

using namespace std;
using namespace cv;

typedef enum DebugLevel
{
	NONE, LOW, MEDIUM, HIGH
} DebugLevel;

int main(int _nargs, char **_vargs)
{
	DebugLevel debugLevel = HIGH;
	bool showImages = false;

	if (_nargs < 2)
	{
		cout << "Not enough arguments\nUsage:\n\tHomework1 <input_file>\n\n";
		return EXIT_FAILURE;
	}

	// Get the calibration matrix
	Mat K = Mat::zeros(3, 3, CV_64FC1);
	Helper::loadCalibrationMatrix(K, _vargs[1]);
	Mat Kt = Mat::zeros(3, 3, CV_64FC1);
	transpose(K, Kt);

	cout << "K:\n";
	Helper::printMatrix<double>(K, 3);

	// Load images
	vector<Mat> images;
	Helper::loadInput(images, _vargs[1]);

	// Get ground truth
	vector<Mat> groundTruth;
	Helper::loadGroundTruth(groundTruth, _vargs[1], images.size());

	// Print groundtruth trajectory
	Printer::getInstance()->calculateConversionRate(400, 400);
	Helper::printTrajectory(groundTruth, "groundTruth.png", RED);

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
//				SVD decompA = SVD(A, SVD::FULL_UV);
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
					Helper::printMatrix<double>(F, 5, "F:");

				Mat E = Kt * F * K;
				if (debugLevel >= MEDIUM)
					Helper::printMatrix<double>(E, 5, "E:");

				SVD decompE = SVD(E, SVD::FULL_UV);

				Mat W = Mat::zeros(3, 3, CV_64FC1);
				W.at<double>(0, 1) = -1;
				W.at<double>(1, 0) = 1;
				W.at<double>(2, 2) = 1;
				Mat R1 = decompE.u * W * decompE.vt;
				Mat R2 = decompE.u * W.t() * decompE.vt;

				if (debugLevel >= HIGH)
				{
					Helper::printMatrix<double>(decompE.u.col(2), 5, "U2:");
					Helper::printMatrix<double>(R1, 5);
					Helper::printMatrix<double>(R2, 5);
				}

				Mat pose = trajectory.back().clone();
				pose.col(3) += decompE.u.col(2);
				trajectory.push_back(pose);

				if (debugLevel >= LOW)
					Helper::printMatrix<double>(trajectory.back(), 5, "Pose:");
			}
		}
	}

	Helper::printTrajectory(trajectory, "trajectory.png", GREEN);

	return EXIT_SUCCESS;
}
