#include <stdlib.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include "Helper.h"

using namespace std;
using namespace cv;

int main(int _nargs, char **_vargs)
{
	bool debug = true;

	vector<Mat> images;
	images.push_back(imread("./input/c0_img_00617_c0_1311874730447572us.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	images.push_back(imread("./input/c0_img_00618_c0_1311874730514324us.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	images.push_back(imread("./input/c0_img_00619_c0_1311874730647599us.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	images.push_back(imread("./input/c0_img_00620_c0_1311874730714324us.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	images.push_back(imread("./input/c0_img_00621_c0_1311874730780951us.jpg", CV_LOAD_IMAGE_GRAYSCALE));

	Ptr<FeatureDetector> featureExtractor = FeatureDetector::create("HARRIS");
	Ptr<DescriptorExtractor> descriptorExtractor = DescriptorExtractor::create("SIFT");
	FlannBasedMatcher matcher;

	vector<vector<KeyPoint>> keypoints;
	vector<Mat> descriptors;

	initModule_nonfree();
	for (size_t k = 0; k < images.size(); k++)
	{
		Mat image = images[k];

		cout << "Extracting keypoints and features for image " << k << "\n";
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

			if (debug)
				Helper::showMatches(images[k - 1], images[k], keypoints[k - 1], keypoints[k], matches);

//			circle(images[k - 1], keypoints[k - 1][goodMatches[0].queryIdx].pt, 5, Scalar(200), 2, 8, 0);
//			namedWindow("Img1", CV_WINDOW_AUTOSIZE);
//			imshow("Img1", images[k - 1]);
//			waitKey(0);
//
//			circle(images[k], keypoints[k][goodMatches[0].trainIdx].pt, 5, Scalar(200), 2, 8, 0);
//			namedWindow("Img2", CV_WINDOW_AUTOSIZE);
//			imshow("Img2", images[k]);
//			waitKey(0);
		}
	}

//	Mat A = Mat::ones(3, 4, CV_32FC1);
//	A.at<float>(0, 0) = 1;
//	A.at<float>(0, 1) = 0;
//	A.at<float>(0, 2) = -8;
//	A.at<float>(0, 3) = -7;
//	A.at<float>(1, 0) = 0;
//	A.at<float>(1, 1) = 1;
//	A.at<float>(1, 2) = 4;
//	A.at<float>(1, 3) = 3;
//	A.at<float>(2, 0) = 0;
//	A.at<float>(2, 1) = 0;
//	A.at<float>(2, 2) = 0;
//	A.at<float>(2, 3) = 0;
//
//	//cout << "det(A): " << determinant(A) << "\n";
//
//	cout << "\nA:\n";
//	Helper::printMatrix<float>(A);
//
//	Mat W = Mat::ones(3, 3, CV_32FC1);
//	Mat U = Mat::ones(3, 3, CV_32FC1);
//	Mat Vt = Mat::ones(3, 3, CV_32FC1);
//	SVD::compute(A, W, U, Vt, SVD::FULL_UV);
//
//	cout << "\nA:\n";
//	Helper::printMatrix<float>(A);
//	cout << "\nU:\n";
//	Helper::printMatrix<float>(U);
//	cout << "\nW:\n";
//	Helper::printMatrix<float>(W);
//	cout << "\nVt:\n";
//	Helper::printMatrix<float>(Vt);

	return EXIT_SUCCESS;
}
