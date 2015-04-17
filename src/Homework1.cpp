#include <stdlib.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "opencv2/nonfree/features2d.hpp"
#include <opencv2/nonfree/nonfree.hpp>
#include "Helper.h"

using namespace std;
using namespace cv;

int main(int _nargs, char **_vargs)
{
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
			vector<DMatch> matches;
			matcher.match(descriptors[k - 1], descriptors[k], matches);

			// Calculation of max and min distances between keypoints
			double max_dist = 0;
			double min_dist = 100;
			for (int i = 0; i < descriptors[k - 1].rows; i++)
			{
				double dist = matches[i].distance;
				min_dist = dist < min_dist ? dist : min_dist;
				max_dist = dist > max_dist ? dist : max_dist;
			}

			// Extract good matches
			vector<DMatch> goodMatches;
			for (int i = 0; i < descriptors[k - 1].rows; i++)
			{
				if (matches[i].distance <= max(2 * min_dist, 0.02))
					goodMatches.push_back(matches[i]);
			}

			// Draw only good matches
			Mat imgMatches;
			drawMatches(images[k - 1], keypoints[k - 1], images[k], keypoints[k], goodMatches, imgMatches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

			// Show detected matches
			namedWindow("Good Matches", CV_WINDOW_AUTOSIZE);
			imshow("Good Matches", imgMatches);
			waitKey(0);

//			Point p1 = keypoints[k - 1][goodMatches[0].queryIdx];
//			circle(images[k - 1], p1, 5, Scalar(200), 2, 8, 0);
//			namedWindow("Img1", CV_WINDOW_AUTOSIZE);
//			imshow("Img1", images[k - 1]);
//			waitKey(0);
//
//			namedWindow("Img2", CV_WINDOW_AUTOSIZE);
//			imshow("Img2", imgMatches);
//			waitKey(0);
		}
	}

//	Mat src, src_gray;
//	int thresh = 150;
//
//	vector <Mat> harrisDescriptor;
//	for (Mat image : images)
//	{
//		Mat dst, dst_norm, dst_norm_scaled;
//		dst = Mat::zeros(image.size(), CV_32FC1);
//
//		// Detector parameters
//		int blockSize = 2;
//		int apertureSize = 3;
//		double k = 0.04;
//
//		// Detecting corners
//		cornerHarris(image, dst, blockSize, apertureSize, k, BORDER_DEFAULT);
//
//		// Normalizing
//		normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
//		convertScaleAbs(dst_norm, dst_norm_scaled);
//		Mat noCircle = dst_norm_scaled.clone();
//
//		// Drawing a circle around corners
//		for (int j = 0; j < dst_norm.rows; j++)
//		{
//			for (int i = 0; i < dst_norm.cols; i++)
//			{
//				if ((int) dst_norm.at<float>(j, i) > thresh)
//				{
//					circle(dst_norm_scaled, Point(i, j), 5, Scalar(200), 2, 8, 0);
//				}
//			}
//		}
//
//		// Showing the result
//		namedWindow("Corners", CV_WINDOW_AUTOSIZE);
//		imshow("Corners", noCircle);
//		waitKey(0);
//
//		namedWindow("Circles", CV_WINDOW_AUTOSIZE);
//		imshow("Circles", dst_norm_scaled);
//		waitKey(0);
//	}
//
//	cout << "Finished!\n";
//
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
