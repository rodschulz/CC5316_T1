/**
 * Author: rodrigo
 * 2015
 */
#include "Helper.h"
#include "Printer.h"
#include <opencv2/highgui/highgui.hpp>
#include <queue>
#include <algorithm>

Helper::Helper()
{
}

Helper::~Helper()
{
}

vector<DMatch> Helper::getMatches(const Mat &_queryDescriptor, const Mat &_trainDescriptor, const FlannBasedMatcher &_matcher)
{
	vector<vector<DMatch>> matches;
	vector<DMatch> goodMatches;
	vector<float> distances;

	_matcher.knnMatch(_queryDescriptor, _trainDescriptor, matches, 2);
	for (size_t i = 0; i < matches.size(); i++)
	{
		if (matches[i][0].distance < 0.5 * matches[i][1].distance)
		{
			goodMatches.push_back(matches[i][0]);
			distances.push_back(matches[i][0].distance);
		}
	}

//	vector<DMatch> matches;
//	vector<DMatch> goodMatches;
//	vector<float> distances;
//
//	_matcher.match(_queryDescriptor, _trainDescriptor, matches);
//	std::sort(matches.begin(), matches.end(), &compareMatches);
//	for (size_t i = 0; i < 30 && i < matches.size(); i++)
//	{
//		goodMatches.push_back(matches[i]);
//		distances.push_back(matches[i].distance);
//	}

	Scalar meanDist, stdDev;
	meanStdDev(distances, meanDist, stdDev);
	cout << "Matches => mean:" << meanDist[0] << " - stdDev:" << stdDev[0] << "\n";

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
	double x1 = _keypoint1.pt.x;
	double y1 = _keypoint1.pt.y;
	double x2 = _keypoint2.pt.x;
	double y2 = _keypoint2.pt.y;

	_A.at<double>(_rowIndex, 0) = x1 * x2;
	_A.at<double>(_rowIndex, 1) = y1 * x2;
	_A.at<double>(_rowIndex, 2) = x2;
	_A.at<double>(_rowIndex, 3) = x1 * y2;
	_A.at<double>(_rowIndex, 4) = y1 * y2;
	_A.at<double>(_rowIndex, 5) = y2;
	_A.at<double>(_rowIndex, 6) = x1;
	_A.at<double>(_rowIndex, 7) = y1;
	_A.at<double>(_rowIndex, 8) = 1;
}

void Helper::getFundamentalMatrix(Mat &_F, const vector<DMatch> &_matches, const vector<KeyPoint> &_keypointsQuery, const vector<KeyPoint> &_keypointsTrain)
{
	vector<Point2f> points1, points2;
	for (DMatch m : _matches)
	{
		points1.push_back(_keypointsQuery[m.queryIdx].pt);
		points2.push_back(_keypointsTrain[m.trainIdx].pt);
	}

	Mat aux = findFundamentalMat(points1, points2, FM_RANSAC, 3, 0.99);
	SVD decomposition(aux);

	Mat diag = Mat::eye(3, 3, CV_64FC1);
	diag.at<double>(2, 2) = 0;

	_F = decomposition.u.clone() * diag * decomposition.vt.clone();
}

void Helper::getEssentialMatrix(Mat &_E, const Mat &_K, const Mat &_F)
{
	// Calculate E (not sure why K gets modified by this, so got to clone it to avoid it)
	Mat E = Mat::zeros(3, 3, CV_64FC1);
	Mat tempK = _K.clone();
	Mat Kt = tempK.t();
	E = Kt * _F * tempK;

	SVD decomposition(E, SVD::FULL_UV);
	Mat diag = Mat::zeros(3, 3, CV_64FC1);
	diag.at<double>(0, 0) = decomposition.w.at<double>(0, 0);
	diag.at<double>(1, 1) = decomposition.w.at<double>(1, 0);

	// Calculate the projected matrix
	_E = decomposition.u * diag * decomposition.vt;
}

void Helper::calculateMotion(const Mat &_E, const Mat &_K, Mat &_R, Mat &_t, Mat &_points3D, const vector<KeyPoint> &_queryKeypoints, const vector<KeyPoint> &_trainKeypoints, const vector<DMatch> &_matches)
{
	Mat W = Mat::zeros(3, 3, CV_64FC1);
	W.at<double>(0, 1) = -1;
	W.at<double>(1, 0) = 1;
	W.at<double>(2, 2) = 1;

	Mat Z = Mat::zeros(3, 3, CV_64FC1);
	Z.at<double>(0, 1) = 1;
	Z.at<double>(1, 0) = -1;

	SVD decomposition(_E);

	Mat S = decomposition.u * Z * decomposition.u.t();
	Mat R1 = decomposition.u * W.t() * decomposition.vt;
	Mat R2 = decomposition.u * W * decomposition.vt;

	// Keep rotations feasible
	if (determinant(R1) < 0)
		R1 = -R1;
	if (determinant(R2) < 0)
		R2 = -R2;

	Mat tx = Mat(3, 1, CV_64FC1);
	tx.at<double>(0, 0) = S.at<double>(2, 1);
	tx.at<double>(1, 0) = S.at<double>(0, 2);
	tx.at<double>(2, 0) = S.at<double>(1, 0);

	vector<pair<Mat, Mat>> motion;
	motion.push_back(make_pair(R1, tx));
	motion.push_back(make_pair(R1, -tx));
	motion.push_back(make_pair(R2, tx));
	motion.push_back(make_pair(R2, -tx));

	// Find the correct pair (triangulation)
//	Mat points;
//	int maxInliers = 0;
//	for (pair<Mat, Mat> p : motion)
//	{
//		int inliers = triangulatePoints1(_queryKeypoints, _trainKeypoints, _K, p.first, p.second, points);
//		if (inliers > maxInliers)
//		{
//			maxInliers = inliers;
//			_points3D = points;
//			_R = p.first;
//			_t = p.second;
//		}
//	}

	// Extract some points for the triangulation
	size_t pointNumber = _matches.size();
	Mat pointsQuery(2, pointNumber, CV_64FC1);
	Mat pointsTrain(2, pointNumber, CV_64FC1);
	for (size_t i = 0; i < pointNumber; i++)
	{
		pointsQuery.at<double>(0, i) = _queryKeypoints[_matches[i].queryIdx].pt.x;
		pointsQuery.at<double>(1, i) = _queryKeypoints[_matches[i].queryIdx].pt.y;
		pointsTrain.at<double>(0, i) = _trainKeypoints[_matches[i].trainIdx].pt.x;
		pointsTrain.at<double>(1, i) = _trainKeypoints[_matches[i].trainIdx].pt.y;
	}

	Mat Pquery = Mat::zeros(3, 4, CV_64FC1);
	Pquery.at<double>(0, 0) = 1;
	Pquery.at<double>(1, 1) = 1;
	Pquery.at<double>(2, 2) = 1;
	Pquery = _K.clone() * Pquery;

	int pointsInFront = -1;
	for (pair<Mat, Mat> entry : motion)
	{
		Mat Ptrain = Mat::zeros(3, 4, CV_64FC1);
		entry.first.col(0).copyTo(Ptrain.col(0));
		entry.first.col(1).copyTo(Ptrain.col(1));
		entry.first.col(2).copyTo(Ptrain.col(2));
		entry.second.col(0).copyTo(Ptrain.col(3));
		Ptrain = _K.clone() * Ptrain;

		Mat points3D;
		triangulatePoints(Pquery, Ptrain, pointsQuery, pointsTrain, points3D);
		//Printer::printMatrix<double>(points3D, 2, "proyection");

		int pointCount = 0;
		for (int i = 0; i < points3D.cols; i++)
			if (points3D.at<double>(2, i) / points3D.at<double>(3, i) > 0)
				pointCount++;

		if (pointCount > pointsInFront)
		{
			pointsInFront = pointCount;
			entry.first.copyTo(_R);
			entry.second.copyTo(_t);
		}
	}

	cout << "Selected transformation with " << pointsInFront << "/" << pointNumber << "\n";
}

int Helper::triangulatePoints(const vector<KeyPoint> &_queryKeypoints, const vector<KeyPoint> &_trainKeypoints, const Mat &_K, const Mat &_R, const Mat &_t, Mat &_points3D)
{
	// Initialize 3d point matrix
	_points3D = Mat(4, _queryKeypoints.size(), CV_64FC1);

	// Projection matrices
	Mat P1 = Mat::zeros(3, 4, CV_64FC1);
	Mat P2 = Mat::zeros(3, 4, CV_64FC1);
	P1.at<double>(0, 0) = _K.at<double>(0, 0);
	P1.at<double>(1, 0) = _K.at<double>(1, 0);
	P1.at<double>(2, 0) = _K.at<double>(2, 0);
	P1.at<double>(0, 1) = _K.at<double>(0, 1);
	P1.at<double>(1, 1) = _K.at<double>(1, 1);
	P1.at<double>(2, 1) = _K.at<double>(2, 1);
	P1.at<double>(0, 2) = _K.at<double>(0, 2);
	P1.at<double>(1, 2) = _K.at<double>(1, 2);
	P1.at<double>(2, 2) = _K.at<double>(2, 2);

	P2.at<double>(0, 0) = _R.at<double>(0, 0);
	P2.at<double>(1, 0) = _R.at<double>(1, 0);
	P2.at<double>(2, 0) = _R.at<double>(2, 0);
	P2.at<double>(0, 1) = _R.at<double>(0, 1);
	P2.at<double>(1, 1) = _R.at<double>(1, 1);
	P2.at<double>(2, 1) = _R.at<double>(2, 1);
	P2.at<double>(0, 2) = _R.at<double>(0, 2);
	P2.at<double>(1, 2) = _R.at<double>(1, 2);
	P2.at<double>(2, 2) = _R.at<double>(2, 2);
	P2.at<double>(0, 3) = _t.at<double>(0, 0);
	P2.at<double>(1, 3) = _t.at<double>(1, 0);
	P2.at<double>(2, 3) = _t.at<double>(2, 0);

	P2 = _K.clone() * P2;

	// Triangulation via orthogonal regression
	Mat J = Mat::zeros(4, 4, CV_64FC1);
	Mat U, S, V;
	for (size_t i = 0; i < _queryKeypoints.size(); i++)
	{
		for (int j = 0; j < 4; j++)
		{
			J.at<double>(0, j) = P1.at<double>(2, j) * _queryKeypoints[i].pt.x - P1.at<double>(0, j);
			J.at<double>(1, j) = P1.at<double>(2, j) * _queryKeypoints[i].pt.y - P1.at<double>(1, j);
			J.at<double>(2, j) = P2.at<double>(2, j) * _trainKeypoints[i].pt.x - P2.at<double>(0, j);
			J.at<double>(3, j) = P2.at<double>(2, j) * _trainKeypoints[i].pt.y - P2.at<double>(1, j);
		}

		SVD decompJ(J);
		Mat v = decompJ.vt.t();
		_points3D.at<double>(0, i) = v.at<double>(0, 3);
		_points3D.at<double>(1, i) = v.at<double>(1, 3);
		_points3D.at<double>(2, i) = v.at<double>(2, 3);
		_points3D.at<double>(3, i) = v.at<double>(3, 3);
	}

	// Calculate inliers
	Mat AX1 = P1 * _points3D;
	Mat BX1 = P2 * _points3D;
	int num = 0;
	for (int i = 0; i < _points3D.cols; i++)
		if (AX1.at<double>(2, i) * _points3D.at<double>(3, i) > 0 && BX1.at<double>(2, i) * _points3D.at<double>(3, i) > 0)
			num++;

	return num;
}

void Helper::extractPoints(const vector<KeyPoint> &_trainKeypoints, const vector<KeyPoint> &_queryKeypoints, const vector<DMatch> &_matches, vector<Point2f> &_trainPoints, vector<Point2f> &_queryPoints)
{
	_trainPoints.clear();
	_queryPoints.clear();
	_trainPoints.reserve(_matches.size());
	_queryPoints.reserve(_matches.size());

	for (size_t i = 0; i < _matches.size(); i++)
	{
		_queryPoints.push_back(_queryKeypoints[_matches[i].queryIdx].pt);
		_trainPoints.push_back(_trainKeypoints[_matches[i].trainIdx].pt);
	}
}

bool Helper::compareMatches(const DMatch &_match1, const DMatch &_match2)
{
	return _match1.distance < _match2.distance;
}
