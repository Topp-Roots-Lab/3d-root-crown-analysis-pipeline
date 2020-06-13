/*****************************/
/*Created on Nov 29, 2018*****/
/*
/*@author: njiang
/*****************************/

#include <iostream>
#include <fstream>
#include <thread>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

namespace fs = boost::filesystem;
namespace po = boost::program_options;

using namespace cv;
using namespace std;

/*
 * Get version number from VERSION file
 * 
 * @param void
 * @returns string semantic version number
 */
string fetchVersion()
{
	// Determine relative location of VERSION file
	string application_filepath;
	char buffer[PATH_MAX];
	ssize_t len = ::readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
	if (len != -1)
	{
		buffer[len] = '\0';
		application_filepath.assign(string(buffer));
	}
	else
	{
		cerr << "Unable to determine file location for application." << endl;
		exit(1);
	}

	// Fetch version from VERSION file
	ifstream ifp;
	fs::path directory_path(application_filepath);
	fs::path relative_filepath("../VERSION");
	fs::path version_realpath = directory_path.parent_path() / relative_filepath;

	string version;
	ifp.open(version_realpath.string());
	string line;
	if (ifp.is_open())
	{
		getline(ifp, line);
		version.assign(line);
	}
	else
	{
		cerr << "Unable to open VERSION file." << endl;
		exit(1);
	}
	ifp.close();
	return version;
}

/* 
 * Calculate median value of matrix
 * 
 * The matrix is a 2-D grayscale image
 * 
 * @param cv::Mat 2-D matrix of unsigned integer values
 * @return single integer value as the median
 */
int medianMat(cv::Mat image)
{
	image = image.reshape(0, 1); // spread image Mat to single row
	std::vector<int> vecFromMat;
	image.copyTo(vecFromMat); // Copy image Mat to vector vecFromMat
	std::nth_element(vecFromMat.begin(), vecFromMat.begin() + vecFromMat.size() / 2, vecFromMat.end());
	return vecFromMat[vecFromMat.size() / 2];
}

/*
 * Convert grayscale images to binary images
 * 
 * 
 * 
 */
int segment()
{
}

int main(int argc, char **argv)
{
	// Configure program options
	try
	{
		// Required parameters
		int doRemove;			 // soil removal flag
		string inputpath;	 // grayscale image directory
		int sampling;			 // downsampling factor (i.e., a sampling of 2 processes half the total grayscale slices)
		string outputpath; // binary image directory
		char *fname2;			 // OUT output filepath
		char *fname3;			 // OBJ output filepath

		string VERSION;
		VERSION.assign(fetchVersion());
	}
	catch (exception &e)
	{
		cerr << e.what() << endl;
	}

	int doRemove = atoi(argv[1]);
	string inputpath = argv[2];
	int sampling = atoi(argv[3]);
	string outputpath = string(argv[4]);
	char *fname2 = argv[5];
	char *fname3 = argv[6];

	FILE *Outfp = fopen(fname2, "w");
	int numVert = 0;

	fprintf(Outfp, "# v%s\n", VERSION.c_str());
	fprintf(Outfp, "%20d\n", numVert);

	FILE *Objfp = fopen(fname3, "w");

	string filePath = string(inputpath) + "*.png";
	vector<String> fn;
	glob(filePath, fn);
	Mat temp = imread(fn[0], CV_LOAD_IMAGE_GRAYSCALE);

	float scale = 1.0 / sampling;
	resize(temp, temp, Size(), scale, scale, INTER_LINEAR);
	int rows, cols, size;
	rows = temp.rows;
	cols = temp.cols;

	size = rows * cols;
	uchar *im = new uchar[size];
	uchar *im2 = new uchar[size];

	memset(temp.data, 0, size);
	double thres1, thres2;
	int count_cur, count_prev = 0, count_prev2 = 0;
	int id;
	float ovlp, ovlp2, ovlp3;
	bool flag = false;

	if (doRemove > 0)
	{
		char *fname4 = argv[7];
		char *fname5 = argv[8];
		FILE *Outfp2 = fopen(fname4, "w");
		int numVert2 = 0;

		fprintf(Outfp2, "# v%s\n", VERSION.c_str());
		fprintf(Outfp2, "%20d\n", numVert);

		FILE *Objfp2 = fopen(fname5, "w");

		for (int n = 0; n < fn.size(); n += sampling)
		//for (int n = fn.size()-1; n >= 0; n -= sampling)
		{
			id = fn.size() - n;
			Mat img = imread(fn[n], CV_LOAD_IMAGE_GRAYSCALE);
			resize(img, img, Size(), scale, scale, INTER_LINEAR);
			medianBlur(img, img, 3);
			Mat bw1, bw2, bw3, bw4;
			memset(im2, 0, size);

			//bw1 -- soil potential
			thres1 = threshold(img, bw1, 0, 255, THRESH_OTSU);
			//bwRemove(bw1, 10);
			//bw2 -- root
			thres2 = threshold(img, bw2, 0, 255, THRESH_TRIANGLE);
			count_cur = countNonZero(bw2);

			if (count_prev > 0 && (thres1 - thres2 > 0))
			{
				//temp -- root image from prevouis slice
				bitwise_and(bw1, temp, bw3);
				ovlp = countNonZero(bw3);
				ovlp2 = ovlp;
				ovlp = ovlp / countNonZero(bw1);
				ovlp2 = ovlp2 / count_prev;
				subtract(bw2, bw1, bw3);
				ovlp2 = countNonZero(bw3);
				ovlp2 = ovlp2 / count_prev;

				bitwise_and(bw3, temp, bw4);
				ovlp3 = countNonZero(bw4);

				bitwise_and(bw2, temp, bw4);

				// count_prev -- number of root pixels in previous slice
				// count_prev2 -- number of soil pixels in prevous slice
				//bitwise_and(bw);
				//if ((abs(countNonZero(bw3) - count_prev) < abs(countNonZero(bw2) - count_prev) && flag == true) ||
				if ((abs(countNonZero(bw3) - count_prev) < abs(countNonZero(bw2) - count_prev) && count_prev2 > 0) ||
						(thres1 - thres2 >= 5 &&
						 ((count_prev2 > 0 && ovlp < 0.7) || (flag == false && n < 0.8 * fn.size() && ovlp2 > 0.7))))
				{
					memcpy(im2, bw1.data, size);
					bw3.copyTo(bw2);
					flag = true;
				}
			}

			count_cur = countNonZero(bw2);
			if (n > 0.7 * fn.size() && count_cur > 50 * count_prev)
			{
				memset(bw2.data, 0, size);
				count_cur = 0;
				memcpy(im2, bw2.data, size);
			}

			count_prev = count_cur;
			memcpy(bw1.data, im2, size);
			count_prev2 = countNonZero(bw1);
			memcpy(im, bw2.data, size);

			for (int i = 0; i < rows; i++)
				for (int j = 0; j < cols; j++)
				{
					int index = i * cols + j;
					if (im[index] > 0)
					{
						numVert++;
						fprintf(Objfp, "v %d %d %d\n", j, i, id / sampling);
						fprintf(Outfp, "%d %d %d\n", j, i, id / sampling);
					}

					if (im2[index] > 0)
					{
						numVert2++;
						fprintf(Objfp2, "v %d %d %d\n", j, i, id / sampling);
						fprintf(Outfp2, "%d %d %d\n", j, i, id / sampling);
					}
				}

			string filename = fn[n].substr(fn[n].find_last_of("\\") + 1);
			imwrite(outputpath + filename, bw2);
			bw2.copyTo(temp);
		}

		fseek(Outfp2, 0L, SEEK_SET);
		rewind(Outfp2);
		fprintf(Outfp, "# v%s\n", VERSION.c_str());
		fprintf(Outfp2, "%20d\n", numVert2);
	}

	else
	{
		for (int n = 0; n < fn.size(); n += sampling)
		{
			id = fn.size() - n;
			Mat img = imread(fn[n], CV_LOAD_IMAGE_GRAYSCALE);
			resize(img, img, Size(), scale, scale, INTER_LINEAR);
			Mat bw2, bw3, bw4;
			memset(im2, 0, size);

			thres2 = threshold(img, bw2, 0, 255, THRESH_TRIANGLE);

			// NOTE(tparker): Check that the threshold value has not been picked from
			// the darker side of the histogram, as it's very unlikely that a root
			// system would be less dense than the air or medium it was scanned in
			// As a workaround, if the threshold value is picked from the darker side,
			// then replace any values in the image that are darker than the median
			// value with the median
			int median = medianMat(img);

			if (thres2 <= median)
			{
				for (int r = 0; r < img.rows; r++)
				{
					for (int c = 0; c < img.cols; c++)
					{
						if (img.at<uint8_t>(r, c) < (uint8_t)median && (uint8_t)img.at<uint8_t>(r, c) != (uint8_t)0)
						{
							img.at<uint8_t>(r, c) = (uint8_t)median;
						}
					}
				}
				// Redo the threshold with the modified image
				thres2 = threshold(img, bw2, median, 255, THRESH_TRIANGLE);
			}

			count_cur = countNonZero(bw2);
			if (n > 0.7 * fn.size() && count_cur > 50 * count_prev)
			{
				memset(bw2.data, 0, size);
				count_cur = 0;
				memcpy(im2, bw2.data, size);
			}

			count_prev = count_cur;
			memcpy(im, bw2.data, size);

			for (int i = 0; i < rows; i++)
				for (int j = 0; j < cols; j++)
				{
					int index = i * cols + j;
					if (im[index] > 0)
					{
						numVert++;
						fprintf(Objfp, "v %d %d %d\n", j, i, id / sampling);
						fprintf(Outfp, "%d %d %d\n", j, i, id / sampling);
					}
				}

			string filename = fn[n].substr(fn[n].find_last_of("/") + 1);
			imwrite(outputpath + filename, bw2);
			bw2.copyTo(temp);
		}
	}

	fseek(Outfp, 0L, SEEK_SET);
	rewind(Outfp);
	fprintf(Outfp, "# v%s\n", VERSION.c_str());
	fprintf(Outfp, "%20d\n", numVert);

	fclose(Outfp);
	fclose(Objfp);
	delete[] im;
	return 0;
}
