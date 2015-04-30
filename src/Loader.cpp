/**
 * Author: rodrigo
 * 2015
 */
#include "Loader.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <dirent.h>
#include <opencv2/highgui/highgui.hpp>

Loader::Loader()
{
}

Loader::~Loader()
{
}

void Loader::loadInput(vector<Mat> &_destination, const string &_inputFile)
{
	_destination.clear();
	int lineCount = 0;

	string line;
	ifstream file;
	file.open(_inputFile.c_str(), fstream::in);
	if (file.is_open())
	{
		string dataFolder = "";

		while (getline(file, line))
		{
			if (line.empty() || line.at(0) == '#')
				continue;

			lineCount++;
			if (lineCount <= 3)
				continue;

			dataFolder = line;
			break;
		}
		file.close();

		vector<string> filenames;
		getFileList(dataFolder, filenames);

//		size_t n = filenames.size();
		size_t n = 400;
		for (size_t i = 0; i < n; i++)
		{
			_destination.push_back(imread(filenames[i], CV_LOAD_IMAGE_GRAYSCALE));
		}
	}
	else
		cout << "Unable to open input: " << _inputFile << "\n";
}

void Loader::loadCalibrationMatrix(Mat &_K, const string &_inputFile)
{
	bool firstLine = true;
	string format = "";

	string line;
	ifstream file;
	file.open(_inputFile.c_str(), fstream::in);
	if (file.is_open())
	{
		while (getline(file, line))
		{
			if (line.empty() || line.at(0) == '#')
				continue;

			if (firstLine)
			{
				format = line;
				firstLine = false;
			}
			else
			{
				if (format.compare("FORMAT_KITTI") == 0)
				{
					string matrixLine;
					ifstream matrixFile;
					matrixFile.open(line.c_str(), fstream::in);

					if (matrixFile.is_open())
					{
						getline(matrixFile, matrixLine);

						// Parse string line
						vector<string> tokens;
						istringstream iss(matrixLine);
						copy(istream_iterator<string>(iss), istream_iterator<string>(), back_inserter(tokens));

						_K.at<double>(0, 0) = stod(tokens[1]);
						_K.at<double>(0, 1) = stod(tokens[2]);
						_K.at<double>(0, 2) = stod(tokens[3]);
						_K.at<double>(0, 3) = stod(tokens[4]);

						_K.at<double>(1, 0) = stod(tokens[5]);
						_K.at<double>(1, 1) = stod(tokens[6]);
						_K.at<double>(1, 2) = stod(tokens[7]);
						_K.at<double>(1, 3) = stod(tokens[8]);

						_K.at<double>(2, 0) = stod(tokens[9]);
						_K.at<double>(2, 1) = stod(tokens[10]);
						_K.at<double>(2, 2) = stod(tokens[11]);
						_K.at<double>(2, 3) = stod(tokens[12]);

						if (_K.cols == 4)
							_K.at<double>(2, 3) = 1;
					}
					else
						cout << "ERROR: Can't read calibration matrix\n";

					matrixFile.close();
					break;
				}
				else if (format.compare("FORMAT_COLLECTION") == 0)
				{
					string matrixLine;
					ifstream matrixFile;
					matrixFile.open(line.c_str(), fstream::in);

					if (matrixFile.is_open())
					{
						int lineCount = 1;
						while (getline(matrixFile, matrixLine))
						{
							switch (lineCount)
							{
								case 12:
								case 15:
								case 18:
								case 21:
								case 24:
								{
									int begin = matrixLine.find_first_of('>');
									int end = matrixLine.find_first_of('<', begin);
									string number = matrixLine.substr(begin + 1, end - begin - 1);

									switch (lineCount)
									{
										case 12:
											_K.at<double>(0, 0) = stod(number);
										break;
										case 15:
											_K.at<double>(1, 1) = stod(number);
										break;
										case 18:
											_K.at<double>(0, 2) = stod(number);
										break;
										case 21:
											_K.at<double>(1, 2) = stod(number);
										break;
										case 24:
											_K.at<double>(0, 1) = stod(number);
										break;
									}
								}
								break;
							}

							lineCount++;
						}

						if (_K.cols == 4)
							_K.at<double>(2, 3) = 1;
					}
					matrixFile.close();
					break;
				}
			}
		}
		file.close();
	}
	else
		cout << "Unable to open input: " << _inputFile << "\n";
}

void Loader::loadGroundTruth(vector<Mat> &_poses, const string &_inputFile, const int _length)
{
	string format = "";
	int lineCount = 0;

	string line;
	ifstream file;
	file.open(_inputFile.c_str(), fstream::in);
	if (file.is_open())
	{
		while (getline(file, line))
		{
			if (line.empty() || line.at(0) == '#')
				continue;

			lineCount++;
			if (lineCount == 1)
				format = line;
			else if (lineCount == 3)
			{
				if (format.compare("FORMAT_KITTI") == 0)
				{
					int poseCount = 0;
					string poseLine;
					ifstream poseFile;
					poseFile.open(line.c_str(), fstream::in);

					if (poseFile.is_open())
					{
						while (poseCount < _length && getline(poseFile, poseLine))
						{
							// Parse string line
							vector<string> tokens;
							istringstream iss(poseLine);
							copy(istream_iterator<string>(iss), istream_iterator<string>(), back_inserter(tokens));

							Mat pose = Mat::zeros(3, 4, CV_64FC1);
							pose.at<double>(0, 0) = stod(tokens[0]);
							pose.at<double>(0, 1) = stod(tokens[1]);
							pose.at<double>(0, 2) = stod(tokens[2]);
							pose.at<double>(0, 3) = stod(tokens[3]);

							pose.at<double>(1, 0) = stod(tokens[4]);
							pose.at<double>(1, 1) = stod(tokens[5]);
							pose.at<double>(1, 2) = stod(tokens[6]);
							pose.at<double>(1, 3) = stod(tokens[7]);

							pose.at<double>(2, 0) = stod(tokens[8]);
							pose.at<double>(2, 1) = stod(tokens[9]);
							pose.at<double>(2, 2) = stod(tokens[10]);
							pose.at<double>(2, 3) = stod(tokens[11]);

							poseCount++;
							_poses.push_back(pose);
						}
					}

					poseFile.close();
				}
				else
				{
				}
			}

		}
		file.close();
	}
	else
		cout << "Unable to open input: " << _inputFile << "\n";
}

void Loader::getFileList(const string &_folder, vector<string> &_fileList)
{
	DIR *folder;
	struct dirent *epdf;
	_fileList.clear();

	if ((folder = opendir(_folder.c_str())) != NULL)
	{
		while ((epdf = readdir(folder)) != NULL)
		{
			if (strcmp(epdf->d_name, ".") == 0 || strcmp(epdf->d_name, "..") == 0)
				continue;

			_fileList.push_back(_folder + epdf->d_name);
		}
		closedir(folder);
	}
	else
	{
		cout << "ERROR: can't open folder " << _folder << "\n";
	}

	sort(_fileList.begin(), _fileList.end());
}
