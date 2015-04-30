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

int main(int _nargs, char **_vargs)
{
	DebugLevel debugLevel = LOW;
	bool showImages = false;

	// Delete previous results
	system("rm -rf ./output/*");

	if (_nargs < 3)
	{
		cout << "Not enough arguments\nUsage:\n\tHomework1 <input_file>\n\n";
		return EXIT_FAILURE;
	}

	// Initialize printer
	Printer::getInstance()->calculateConversionRate(1000, 1000);

	// Method to be used to solve the visual odometry problem
	string method = _vargs[2];
	cout << "Using method " << method << "\n";

	// Get the calibration matrix
	Mat K = Mat::zeros(3, 3, CV_64FC1);
	Loader::loadCalibrationMatrix(K, _vargs[1]);

	// Load images
	vector<Mat> images;
	Loader::loadInput(images, _vargs[1]);

	// Get ground truth
	vector<Mat> groundTruth;
	Loader::loadGroundTruth(groundTruth, _vargs[1], images.size());

	// Print groundtruth trajectory
	Printer::printTrajectory(groundTruth, "groundTruth.png", RED);

	Ptr<FeatureDetector> featureExtractor = FeatureDetector::create("HARRIS");
	Ptr<DescriptorExtractor> descriptorExtractor = DescriptorExtractor::create("SIFT");
	FlannBasedMatcher matcher;

	vector<vector<KeyPoint>> keypoints;
	vector<Mat> descriptors;
	vector<Mat> trajectory;
	Mat start = Mat::zeros(4, 1, CV_64FC1);
	start.at<double>(0, 3) = 1;
	trajectory.push_back(start);

	// Process data
	initModule_nonfree();
	for (size_t j = 0; j < images.size(); j++)
	{
		cout << "*** Processing image " << j << " ***\n";
		Mat image = images[j];

		// Keypoints extraction
		keypoints.push_back(vector<KeyPoint>());
		featureExtractor->detect(image, keypoints.back());

		// Feature extraction
		descriptors.push_back(Mat());
		descriptorExtractor->compute(image, keypoints.back(), descriptors.back());

		if (j > 0)
		{
			int train = keypoints.size() - 1;
			int query = train - 1;

			// Match points between images
			vector<DMatch> matches = Helper::getMatches(descriptors[query], descriptors[train], matcher);

			if (showImages)
				Helper::showMatches(images[query], images[train], keypoints[query], keypoints[train], matches);

			if (matches.size() >= 8)
			{
				Mat transformation;
				if (method.compare("essential_matrix") == 0)
				{
					/** RESOLUTION USING ESSENTIAL MATRIX */

					// Calculate the fundamental matrix
					Mat F;
					Helper::getFundamentalMatrix(F, matches, keypoints[query], keypoints[train]);
					if (debugLevel >= LOW)
						Printer::printMatrix<double>(F, 3, "F:");

					// Calculate E
					//Mat E;
					//Helper::getEssentialMatrix(E, K, F);
					//if (debugLevel >= MEDIUM)
					//Printer::printMatrix<double>(E, 3, "E:");

					// Calculate the motion between the two images
					Mat R, tx, points3D;
					Helper::calculateMotion(F, K, R, tx, points3D, keypoints[query], keypoints[train], matches);

					hconcat(R, tx, transformation);
				}
				else if (method.compare("homography") == 0)
				{
					/** RESOLUTION USING THE HOMOGRAPHY */
					vector<unsigned char> matchesUsedMask;
					vector<Point2f> trainPoints, queryPoints;
					Helper::extractPoints(keypoints[train], keypoints[query], matches, trainPoints, queryPoints);
					Mat H = findHomography(trainPoints, queryPoints, RANSAC, 4, matchesUsedMask);

					transformation = Mat::eye(3, 4, CV_64FC1);
					double norm1 = (double) norm(H.col(0));
					double norm2 = (double) norm(H.col(1));
					double tnorm = (norm1 + norm2) / 2.0f;

					Mat v1 = H.col(0);
					Mat v2 = transformation.col(0);
					cv::normalize(v1, v2);

					v1 = H.col(1);
					v2 = transformation.col(1);
					cv::normalize(v1, v2);

					v1 = transformation.col(0);
					v2 = transformation.col(1);
					Mat v3 = v1.cross(v2);
					Mat c2 = transformation.col(2);
					v3.copyTo(c2);

					transformation.col(3) = H.col(2) / tnorm;
				}

				Mat pose = transformation * trajectory.back().clone();
				vconcat(pose, Mat::eye(1, 1, CV_64FC1), pose);
				trajectory.push_back(pose);

				if (debugLevel >= LOW)
					Printer::printMatrix<double>(trajectory.back(), 3, "Pose:");
			}
			else
			{
				keypoints.erase(keypoints.begin() + keypoints.size() - 1, keypoints.end());
			}
		}
	}

	Printer::getInstance()->calculateConversionRate(400, 400);

	cout << "Printing final trajectory\n";
	Printer::printTrajectory2(trajectory, "trajectory.png", GREEN);

	return EXIT_SUCCESS;
}
