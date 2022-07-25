/*****************************/
/*Created on Nov 29, 2018*****/
/*
/*@author: njiang, tparker
/*****************************/

#include <glob.h>
#include <tbb/tbb.h>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <fstream>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace fs = boost::filesystem;
namespace po = boost::program_options;

using namespace cv;
using namespace std;

const string MODULE_NAME = "rootCrownSegmentation";
const string VERSION_NO = "1.2.0";
const string VERSION = MODULE_NAME + " " + VERSION_NO;

/*
 * Calculate median value of matrix
 *
 * The matrix is a 2-D grayscale image
 *
 * @param cv::Mat 2-D matrix of unsigned integer values
 * @return single integer value as the median
 */
int medianMat(cv::Mat image) {
  image = image.reshape(0, 1);  // spread image Mat to single row
  std::vector<int> vecFromMat;
  image.copyTo(vecFromMat);  // Copy image Mat to vector vecFromMat
  std::nth_element(vecFromMat.begin(),
                   vecFromMat.begin() + vecFromMat.size() / 2,
                   vecFromMat.end());
  return vecFromMat[vecFromMat.size() / 2];
}

/*
 * Scales a value from one range to another range, inclusive.
 *
 * This functions uses globally assigned values, min and max, of N given .nsidat
 * files
 *
 * @param x value to be transformed
 * @param a minimum of input range
 * @param b maximum of input range
 * @param c minimum of output range
 * @param d maximum of output range
 * @return equivalent value of the input value within a new target
 * range
 */
uint8_t scale(uint8_t x, uint8_t a, uint8_t b, uint8_t c, uint8_t d) {
  printf("Scaling %hhu\n", x);
  return (uint8_t)((x - a) / (float)(b - a) * (d - c) + c);
}

/*
 * Calculate mode of a vector
 *
 * The vector is a 1-D list of numeric values
 *
 * @param std::vector<uint8_t> list
 * @return uint8_t mode
 */
uint8_t mode(std::vector<uint8_t> &arr) {
  std::vector<uint8_t> histogram(pow(2, 16), 0);
  for (int i = 0; i < arr.size(); ++i) ++histogram[arr[i]];
  return std::max_element(histogram.begin(), histogram.end()) -
         histogram.begin();
}

/*
 * Calculate standard of deviation of a vector
 *
 * The vector is a 1-D list of numeric values
 *
 * @param std::vector<uint8_t> list
 * @return float standard of deviation
 */
float stddev(std::vector<uint8_t> &arr) {
  float sum{};
  float mean;
  float standardDeviation{};

  int i;
  for (i = 0; i < 10; ++i) {
    sum += arr[i];
  }

  mean = sum / arr.size();

  for (i = 0; i < 10; ++i) {
    standardDeviation += pow(arr[i] - mean, 2);
  }

  return sqrt(standardDeviation / arr.size());
}

/*
 * Check for circular artifact in a slice
 *
 * The matrix is a
 *
 * @param cv::Mat* binary image
 * @return `true` or `false`, whether a circle artifact is detected
 */
bool hasCircularArtifact(int slice_index, cv::Mat &data) {
  // Base case: no image
  if (data.empty()) {
    return false;
  }

  // Make a temporary copy of the binary image to work on
  cv::Mat img = cv::Mat(data.size(), data.type());
  data.copyTo(img);

  cv::Size shape = img.size();
  const int width = shape.width;
  const int height = shape.height;
  const int depth = img.depth();
  cv::Point center = cv::Point(width / 2, height / 2);

  const int minRadius = ((std::min(width, height)) / 15);
  const int maxRadius = ((std::min(width, height)) / 4) - 1;
  // For each possible radius of a circle on the image, from largest to
  // smallest, check proportion of white pixels
  // std::cout << "Min/max radii: [" << minRadius << ", " << maxRadius << "]" <<
  // std::endl; std::cout << "Radius,Pixel Count (src + inner),Pixel Count (mask
  // + inner),Ratio (inner),Pixel Count (src + outer),Pixel Count (mask +
  // outer),Ratio (outer)" << std::endl;

  // Decrementing value used to search for circular artifacts instead
  // of just search 1 by 1 resulting in a O(n) runtime.
  int radius;
  int last_checked_radius = maxRadius;
  int searchRate = 1;
  // Key that was pressed while waiting for cv::waitKey() calls
  int keyCode;
  for (radius = minRadius; radius <= maxRadius; radius += minRadius) {
    // Inner mask
    cv::Mat mask = cv::Mat::zeros(img.size(), img.type());
    cv::circle(mask, center, radius, cv::Scalar(255, 255, 255), CV_FILLED);
    const int mask_white_pixel_count = cv::countNonZero(mask);

    // Outer mask
    cv::Mat inverted_mask = cv::Mat::zeros(img.size(), img.type());
    cv::bitwise_not(mask, inverted_mask);
    const int inverted_mask_white_pixel_count = cv::countNonZero(inverted_mask);

    // Copy input matrix's contents within inner mask
    // Binary representation of
    cv::Mat masked_binary_image = cv::Mat::zeros(img.size(), img.type());

    img.copyTo(masked_binary_image, mask);
    // Copy input matrix's contents within outer mask
    cv::Mat inverted_masked_binary_image =
        cv::Mat::zeros(img.size(), img.type());
    img.copyTo(inverted_masked_binary_image, inverted_mask);

    // Gather white pixels within inner ROI and dilate them
    // This is to accecuate the existant of a circle

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::Mat tmp =
        cv::Mat(masked_binary_image.size(), masked_binary_image.type());
    cv::dilate(masked_binary_image, tmp, kernel, cv::Point(-1, -1), 3);
    tmp.copyTo(masked_binary_image);

    const int masked_binary_image_white_pixel_count =
        cv::countNonZero(masked_binary_image);
    const int inverted_masked_binary_image_white_pixel_count =
        cv::countNonZero(inverted_masked_binary_image);
    double inner_mask_white_pixel_ratio =
        (double(masked_binary_image_white_pixel_count) /
         mask_white_pixel_count);
    double outer_mask_white_pixel_ratio =
        (double(inverted_masked_binary_image_white_pixel_count) /
         inverted_mask_white_pixel_count);

    // printf("%d,%d,%d,%d,%0.13f,%d,%d,%0.13f\n", slice_index, radius,
    // masked_binary_image_white_pixel_count, mask_white_pixel_count,
    // inner_mask_white_pixel_ratio,
    // inverted_masked_binary_image_white_pixel_count,
    // inverted_mask_white_pixel_count, outer_mask_white_pixel_ratio);

    if (inner_mask_white_pixel_ratio < 0.05) {
      return false;
    }

    // Check for a reasonably dense and high proportion of white pixels
    // given a radius of around the center of the image
    // if (inner_mask_white_pixel_ratio > 0.20 && outer_mask_white_pixel_ratio <
    // 0.10)
    if (inner_mask_white_pixel_ratio > 0.80) {
      // printf("%d,%d,%d,%d,%0.13f,%d,%d,%0.13f\n", slice_index, radius,
      // masked_binary_image_white_pixel_count, mask_white_pixel_count,
      // inner_mask_white_pixel_ratio,
      // inverted_masked_binary_image_white_pixel_count,
      // inverted_mask_white_pixel_count, outer_mask_white_pixel_ratio);
      return true;
    }
  }
  // Otherwise, it's not reasonable to assume that any flagged pixels are
  // make up a circular artifact
  return false;
}

/*
 * Convert grayscale images to binary images
 *
 * @param grayscale_images_directory Filepath to grayscale images directory
 * @param sampling Downsampling factor (e.g., 2 equates to 1/(2^3) overall
 * scale)
 * @param binary_images_directory Destination filepath to output directory for
 * binary images
 * @param filepath_out Destination filepath to output OUT file
 * @param filepath_obj Destination filepath to output OBJ file
 * @param lower_bound The lowest value to be *included* during segmentation
 * @param upper_bound The highest value to be *included* during segmentation
 */
int user_defined_segment(string grayscale_images_directory, int sampling,
                         string binary_images_directory, string filepath_out,
                         string filepath_obj, int lowerb = 0,
                         int upperb = 255) {
  // Initialize OUT file
  FILE *Outfp = fopen(filepath_out.c_str(), "w");

  int numVert{};
  fprintf(Outfp, "# %s\n", VERSION.c_str());
  fprintf(Outfp, "%20d\n", numVert);

  // Initialize OBJ file
  FILE *Objfp = fopen(filepath_obj.c_str(), "w");

  // Gather list of all grayscale images
  string filePath = grayscale_images_directory + "*.png";
  vector<String> fn;
  glob(filePath, fn);

  // Set scale based on downsampling factor
  float scale = 1.0 / sampling;  // reciprocal of sampling value
  std::cout << "Scale set as '" << scale << '\n';

  // Use first image to initialize dimensions and memory requirements
  Mat temp = imread(fn[0], CV_LOAD_IMAGE_GRAYSCALE);
  cv::resize(temp, temp, Size(), scale, scale, INTER_LINEAR);
  int rows{temp.rows};
  int cols{temp.cols};
  temp.release();  // free temporary image from memory
  int size{rows * cols};

  int id;  // inverted slice index because roots are usually scanned upside down

  int count_cur, count_prev = 0, count_prev2 = 0;  // white pixel counter(s)

  // For each grayscale image...
  int slice_count = fn.size();
  for (int n = 0; n < slice_count; n += sampling) {
    id = slice_count - n;
    Mat grayscale_image = imread(fn[n], CV_LOAD_IMAGE_GRAYSCALE);
    resize(grayscale_image, grayscale_image, Size(), scale, scale,
           INTER_LINEAR);
    Mat binary_image;  // thresholded image data

    cv::inRange(grayscale_image, lowerb, upperb, binary_image);

    // Write thresholded binary image to disk
    string filename = fn[n].substr(fn[n].find_last_of("/") + 1);
    std::cout << "Slice '" << n
              << "' manually segmented with lower and upper bounds of ["
              << lowerb << ", " << upperb << "] " << filename << std::endl;

    // Convert threshold image to OBJ and OUT files
    for (int i = 0; i < rows; i++)
      for (int j = 0; j < cols; j++) {
        int index = i * cols + j;
        // If the pixel is not black, create a vertex
        if (binary_image.data[index] > 0) {
          numVert++;
          // Vertical index has to be squashed along the vertical axis to
          // compensate for scaling, therefore (id / sampling) = Z position
          fprintf(Objfp, "v %d %d %d\n", j, i, id / sampling);
          fprintf(Outfp, "%d %d %d\n", j, i, id / sampling);
        }
      }
    imwrite(binary_images_directory + filename, binary_image);
    cout << "Write binary image '" << filename << "'" << endl;
  }

  // Update vertex count and version for OUT file
  fseek(Outfp, 0L, SEEK_SET);
  rewind(Outfp);
  fprintf(Outfp, "# %s\n", VERSION.c_str());
  fprintf(Outfp, "%20d\n", numVert);

  // Clean up
  fclose(Outfp);
  fclose(Objfp);
  return 0;
}

/*
 * Convert grayscale images to binary images
 *
 * @param grayscale_images_directory Filepath to grayscale images directory
 * @param sampling Downsampling factor (e.g., 2 equates to 1/(2^3) overall
 * scale)
 * @param binary_images_directory Destination filepath to output directory for
 * binary images
 * @param filepath_out Destination filepath to output OUT file
 * @param filepath_obj Destination filepath to output OBJ file
 */
int segment(string grayscale_images_directory, int sampling,
            string binary_images_directory, string filepath_out,
            string filepath_obj) {
  // Initialize OUT file
  FILE *Outfp = fopen(filepath_out.c_str(), "w");
  int numVert = 0;
  fprintf(Outfp, "# %s\n", VERSION.c_str());
  fprintf(Outfp, "%20d\n", numVert);

  // Initialize OBJ file
  FILE *Objfp = fopen(filepath_obj.c_str(), "w");

  // Gather list of all grayscale images
  string filePath = grayscale_images_directory + "*.png";
  vector<String> fn;
  glob(filePath, fn);

  // Set scale based on downsampling factor
  float scale = 1.0 / sampling;  // reciprocal of sampling value
  cout << "Scale set as '" << scale << "'" << endl;

  // Use first image to initialize dimensions and memory requirements
  Mat temp = imread(fn[0], CV_LOAD_IMAGE_GRAYSCALE);
  resize(temp, temp, Size(), scale, scale, INTER_LINEAR);
  int rows, cols, size;
  rows = temp.rows;
  cols = temp.cols;
  temp.release();  // free temporary image from memory
  size = rows * cols;

  double threshold_value;  // grayscale value selected as threshold value
  int id;  // inverted slice index because roots are usually scanned upside down

  int count_cur, count_prev = 0, count_prev2 = 0;  // white pixel counter(s)

  // For each grayscale image...
  int slice_count = fn.size();
  for (int n = 0; n < slice_count; n += sampling) {
    std::string thresholding_method = "triangle";
    id = slice_count - n;
    Mat grayscale_image = imread(fn[n], CV_LOAD_IMAGE_GRAYSCALE);
    resize(grayscale_image, grayscale_image, Size(), scale, scale,
           INTER_LINEAR);
    Mat binary_image;  // thresholded image data

    threshold_value =
        threshold(grayscale_image, binary_image, 0, 255, CV_THRESH_TRIANGLE);
    // printf("Threshold value: %f\n", threshold_value);

    // NOTE(tparker): Check that the threshold value has not been picked from
    // the darker side of the histogram, as it's very unlikely that a root
    // system would be less dense than the air or medium it was scanned in
    // As a workaround, if the threshold value is picked from the darker side,
    // then replace any values in the image that are darker than the median
    // value with the median
    int median = medianMat(grayscale_image);
    double min, max;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(grayscale_image, &min, &max, &minLoc, &maxLoc);
    bool blankSliceFlag = false;

    // Narrow histogram check
    // If the distance between the median and minimum is greater than the
    // maximum and median by some factor, then the histogram is likely narrow
    // Also, if the distance between the maximum and minimum is less than some
    // constant value, then it is likely a narrow histogram as well
    if ((median / max) > 0.75) {
      count_cur = 0;
      memset(binary_image.data, 0, size);
      blankSliceFlag = true;
    }

    // Check for 'circle' artifact at the upper and lower bounds of the object
    // A circular artifact appears to happen when there is noise near the top
    // or bottom of an object in a scan.
    if (!blankSliceFlag &&
        ((n < (slice_count * 0.05) || n > (slice_count * 0.70)) &&
         hasCircularArtifact(n, binary_image))) {
      // std::cout << "Flagged for circular artifact!" << std::endl;
      int triangle_threshold_value = threshold_value;
      // std::cout << "Triangle Treshold: " << triangle_threshold_value <<
      // std::endl;

      int otsu_threshold_value =
          threshold(grayscale_image, binary_image, 0, 255, THRESH_OTSU);
      // std::cout << "Otsu Treshold: " << otsu_threshold_value << std::endl;

      int adjusted_thresholed_value =
          floor((triangle_threshold_value + otsu_threshold_value) / 2.0);
      adjusted_thresholed_value =
          floor((triangle_threshold_value + adjusted_thresholed_value) / 2.0);
      adjusted_thresholed_value = otsu_threshold_value;

      // In the rare instance that Otsu's method picks a darker threshold,
      // default to the triangle method's result instead, as the higher the
      // threshold, the less aggressive of thresholding.
      if (otsu_threshold_value < triangle_threshold_value) {
        adjusted_thresholed_value = triangle_threshold_value;
      }
      threshold_value =
          threshold(grayscale_image, binary_image, adjusted_thresholed_value,
                    255, THRESH_BINARY);
      // std::cout << "Adjusted Threshold: " << threshold_value << std::endl;

      // TODO(tparker): When the circle is still there!
      // If the circular artifact is still present after
      if (hasCircularArtifact(n, binary_image)) {
        threshold_value = threshold(grayscale_image, binary_image,
                                    otsu_threshold_value, 255, THRESH_BINARY);
        thresholding_method = "Otsu";
      } else {
        thresholding_method = "Adjusted (mean of triangle & otsu)";
      }
    }

    // Get the number of white pixels for the current slice
    count_cur = countNonZero(binary_image);

    // If more than 70% of the slice is consider root system, then it likely
    // that the wrong threshold value was selected and the medium was included
    if (!blankSliceFlag && count_cur > (0.7 * size)) {
      count_cur = 0;
      memset(binary_image.data, 0, size);
      blankSliceFlag = true;
    }

    // Compare the white pixel counts between the current slice and previous
    // slice Reset the thresholded image to pure black when... 70% of the volume
    // has been processed AND The number of white pixels on the current slice is
    // 50 times more than the previous slice
    if (!blankSliceFlag && n > 0.7 * fn.size() && count_cur > 50 * count_prev) {
      count_cur = 0;
      memset(binary_image.data, 0, size);
      blankSliceFlag = true;
    }

    // Write thresholded binary image to disk
    string filename = fn[n].substr(fn[n].find_last_of("/") + 1);
    std::cout << "Slice '" << n << "' segmented with " << thresholding_method
              << " (" << threshold_value << ") for " << filename << std::endl;
    // Update "previous" state to processed "current" state values
    count_prev = count_cur;

    // Convert threshold image to OBJ and OUT files
    for (int i = 0; i < rows; i++)
      for (int j = 0; j < cols; j++) {
        int index = i * cols + j;
        // If the pixel is not black, create a vertex
        if (binary_image.data[index] > 0) {
          numVert++;
          // Vertical index has to be squashed along the vertical axis to
          // compensate for scaling, therefore (id / sampling) = Z position
          fprintf(Objfp, "v %d %d %d\n", j, i, id / sampling);
          fprintf(Outfp, "%d %d %d\n", j, i, id / sampling);
        }
      }
    imwrite(binary_images_directory + filename, binary_image);
    cout << "Write binary image '" << filename << "'" << endl;
  }

  // Update vertex count and version for OUT file
  fseek(Outfp, 0L, SEEK_SET);
  rewind(Outfp);
  fprintf(Outfp, "# %s\n", VERSION.c_str());
  fprintf(Outfp, "%20d\n", numVert);

  // Clean up
  fclose(Outfp);
  fclose(Objfp);
  return 0;
}

/*
 * Convert grayscale images to binary images (w/ soil removal)
 *
 * @param grayscale_images_directory Filepath to grayscale images directory
 * @param sampling Downsampling factor (e.g., 2 equates to 1/(2^3) overall
 * scale)
 * @param binary_images_directory Destination filepath to output directory for
 * binary images
 * @param filepath_out Destination filepath to output OUT file (root system)
 * @param filepath_obj Destination filepath to output OBJ file (root system)
 * @param filepath_out_soil Destination filepath to output OUT file (soil)
 * @param filepath_obj_soil Destination filepath to output OBJ file (soil)
 */
int segment(string grayscale_images_directory, int sampling,
            string binary_images_directory, string filepath_out,
            string filepath_obj, string filepath_out_soil,
            string filepath_obj_soil) {
  // Initialize OUT files
  FILE *Outfp_root =
      fopen(filepath_out.c_str(), "w");  // Root .OUT file handler
  FILE *Outfp_soil =
      fopen(filepath_out_soil.c_str(), "w");  // Soil .OUT file handler

  int numVert_root = 0;
  fprintf(Outfp_root, "# %s\n", VERSION.c_str());
  fprintf(Outfp_root, "%20d\n", numVert_root);
  int numVert_soil = 0;
  fprintf(Outfp_soil, "# %s\n", VERSION.c_str());
  fprintf(Outfp_soil, "%20d\n", numVert_soil);

  // Initialize OBJ files
  FILE *Objfp_root =
      fopen(filepath_obj.c_str(), "w");  // Root .OBJ file handler
  FILE *Objfp_soil =
      fopen(filepath_obj_soil.c_str(), "w");  // Soil .OBJ file handler

  // Gather list of all grayscale images
  string filePath = string(grayscale_images_directory) + "*.png";
  vector<String> fn;  // list of all grayscale image filepaths
  glob(filePath, fn);

  // Set scale based on downsampling factor
  float scale = 1.0 / sampling;  // reciprocal of sampling value
  cout << "Scale set as '" << scale << "'" << endl;

  // Use resized first image to initialize dimensions and memory requirements
  Mat temp = imread(fn[0], CV_LOAD_IMAGE_GRAYSCALE);  // working grayscale image
  resize(temp, temp, Size(), scale, scale, INTER_LINEAR);
  int rows, cols, size;
  rows = temp.rows;
  cols = temp.cols;
  size = rows * cols;

  // Clear temporary image, replacing first image slice in memory
  memset(temp.data, 0, size);

  double soil_threshold_value,
      root_threshold_value;  // grayscale intensity value
  int id;  // inverted slice index because roots are usually scanned upside down

  int count_cur_root,       // number of root pixels in current slice
      count_prev_root = 0,  // number of root pixels in previous slice
      count_prev_soil = 0;  // number of soil pixels in previous slice
  float ovlp, ovlp2, ovlp3;
  bool flag = false;  // ?

  // For each grayscale image...
  for (int n = 0; n < fn.size(); n += sampling) {
    id = fn.size() - n;
    Mat grayscale_image = imread(fn[n], CV_LOAD_IMAGE_GRAYSCALE);
    resize(grayscale_image, grayscale_image, Size(), scale, scale,
           INTER_LINEAR);
    medianBlur(grayscale_image, grayscale_image, 3);

    Mat soil_binary_image,  // soil potential
        root_binary_image,  // root system
        bw3, bw4;

    // Perform auto thresholding
    soil_threshold_value =
        threshold(grayscale_image, soil_binary_image, 0, 255, THRESH_OTSU);
    root_threshold_value =
        threshold(grayscale_image, root_binary_image, 0, 255, THRESH_TRIANGLE);

    // Count the number of root pixels for the current slice for comparison
    count_cur_root = countNonZero(root_binary_image);

    // When the previous slice has root material
    // AND
    // the threshold for current root is less dense/bright than that of the
    // current soil
    if (count_prev_root > 0 &&
        (soil_threshold_value - root_threshold_value > 0)) {
      // temp -> root image from previous slice
      bitwise_and(soil_binary_image, temp,
                  bw3);  // intersection of current soil and previous root
      ovlp = countNonZero(bw3);  // number of white pixels of intersection of
                                 // current soil and previous root
      ovlp2 = ovlp;  // number of white pixels of intersection of current soil
                     // and previous root
      ovlp =
          ovlp /
          countNonZero(
              soil_binary_image);  // percentage -> (intersection of current
                                   // soil and previous root) / (current soil)
      subtract(root_binary_image, soil_binary_image,
               bw3);  // difference between current root and current soil
      ovlp2 = countNonZero(bw3);  // number of white pixels for difference
                                  // between current root and current soil
      ovlp2 =
          ovlp2 / count_prev_root;  // percentage -> (difference of current root
                                    // and current soil) / previous root

      bitwise_and(bw3, temp,
                  bw4);  // intersection of (difference between current root and
                         // current soil) and previous root
      ovlp3 = countNonZero(
          bw4);  // number of white pixels for intersection of (difference
                 // between current root and current soil) and previous root

      bitwise_and(root_binary_image, temp,
                  bw4);  // intersection of current root and previous root

      if (
          // ( |(pxlCount of diff cRoot and cSoil - pxlCount of pRoot)| <
          // |pxlCount of cRoot| ) AND (there was soil in previous slice)
          (abs(countNonZero(bw3) - count_prev_root) <
               abs(countNonZero(root_binary_image) - count_prev_root) &&
           count_prev_soil > 0) ||
          // OR
          // (Soil is more dense than root by 5 units) AND
          // 			(there was soil in previous slice AND the
          // percentage of (intersection of current soil and previous root) /
          // (current soil) is less than 70% ) 			 OR
          //      (At least one slice has been flagged previously)
          //         AND
          //      (the current slice is in the top 80% of volume)
          //         AND
          //      (percentage of (difference of current root and current soil) /
          //      previous root) is greater than 70%)
          (soil_threshold_value - root_threshold_value >= 5 &&
           ((count_prev_soil > 0 && ovlp < 0.7) ||
            (flag == false && n < 0.8 * fn.size() && ovlp2 > 0.7)))) {
        // Replace current root with the difference between current root and
        // current soil
        bw3.copyTo(root_binary_image);
        flag = true;
      }
    }

    // Compare the white pixel counts between the current slice and previous
    // slice
    count_cur_root = countNonZero(root_binary_image);
    // cout << "n: " << n << "\tcount_cur: " << count_cur << "\t fn.size(): " <<
    // fn.size() << "\tcount_prev * 50: " << (50 * count_prev) << "\tCheck
    // result: " << (n > 0.7 * fn.size() && count_cur > 50 * count_prev) <<
    // endl; Reset the thresholded image to pure black when... 70% of the volume
    // has been processed AND The number of white pixels on the current slice is
    // 50 times more than the previous slice
    if (n > 0.7 * fn.size() && count_cur_root > 50 * count_prev_root) {
      count_cur_root = 0;
      memset(root_binary_image.data, 0, size);
    }

    // Update "previous" state to processed "current" state values
    count_prev_root = count_cur_root;
    count_prev_soil = countNonZero(soil_binary_image);

    // Convert threshold images to OBJ and OUT files
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        int index = i * cols + j;
        // If the pixel is not black, create a vertex
        if (root_binary_image.data[index] > 0) {
          numVert_root++;
          // Vertical index has to be squashed along the vertical axis to
          // compensate for scaling, therefore (id / sampling) = Z position
          fprintf(Objfp_root, "v %d %d %d\n", j, i, id / sampling);
          fprintf(Outfp_root, "%d %d %d\n", j, i, id / sampling);
        }

        // If the pixel is not black, create a vertex
        if (soil_binary_image.data[index] > 0) {
          numVert_soil++;
          // Vertical index has to be squashed along the vertical axis to
          // compensate for scaling, therefore (id / sampling) = Z position
          fprintf(Objfp_soil, "v %d %d %d\n", j, i, id / sampling);
          fprintf(Outfp_soil, "%d %d %d\n", j, i, id / sampling);
        }
      }
    }

    // Write thresholded binary image to disk
    string filename = fn[n].substr(fn[n].find_last_of("/") + 1);
    string filepath = binary_images_directory + filename;
    imwrite(filepath, root_binary_image);
    cout << "Write binary image '" << filepath << "'" << endl;

    // Save current root image data to temp for comparison with next slice
    root_binary_image.copyTo(temp);
  }

  // Update vertex count and version for OUT files
  fseek(Outfp_soil, 0L, SEEK_SET);
  rewind(Outfp_soil);
  fprintf(Outfp_soil, "# %s\n", VERSION.c_str());
  fprintf(Outfp_soil, "%20d\n", numVert_soil);

  fseek(Outfp_root, 0L, SEEK_SET);
  rewind(Outfp_root);
  fprintf(Outfp_root, "# %s\n", VERSION.c_str());
  fprintf(Outfp_root, "%20d\n", numVert_root);

  // Clean up
  fclose(Outfp_soil);
  fclose(Outfp_root);
  fclose(Objfp_root);
  fclose(Objfp_soil);
  return 0;
}

int main(int argc, char **argv) {
  try {
    // Configure program options
    // Required parameters
    bool soil_removal_flag;             // soil removal flag
    string grayscale_images_directory;  // grayscale image directory (PNG)
    int sampling;  // downsampling factor (i.e., a sampling of 2 processes half
                   // the total grayscale slices)
    string binary_images_directory;  // binary image directory
    string filepath_out;             // OUT output filepath (root system)
    string filepath_obj;             // OBJ output filepath (root system)
    string filepath_out_soil;        // OUT output filepath (soil)
    string filepath_obj_soil;        // OBJ output filepath (soil)
    bool manual_segmentation;  // manual segmentation flag (use lower and upper
                               // bounds for segmentation)
    int lower_bound;           // Lower bound for user-defined threshold
    int upper_bound;           // Upper bound for user-defined threshold

    po::options_description generic("");
    generic.add_options()("help,h", "show this help message and exit")(
        "version,V", "show program's version number and exit")(
        "verbose,v", "increase output verbosity. (default: False)")(
        "sampling", po::value<int>(&sampling)->default_value(2),
        "downsampling factor")("remove-soil",
                               po::bool_switch(&soil_removal_flag),
                               "enable automatic soil removal")(
        "manual,m", po::bool_switch(&manual_segmentation),
        "enable manual segmentation (you should provide lower and upper "
        "bounds)")(
        "lower-bound,l", po::value<int>(&lower_bound)->default_value(0),
        "set lower bound value to use during segmentation (inclusive)")(
        "upper-bound,u", po::value<int>(&upper_bound)->default_value(255),
        "set upper bound value to use during segmentation (inclusive)");

    po::options_description hidden("Hidden options");
    hidden.add_options()("grayscale-images-directory",
                         "filepath to directory containing grayscale images")(
        "binary-images-directory",
        "filepath to directory to store binary images")(
        "out-filepath", "filepath for produced .OUT file")(
        "obj-filepath", "filepath for produced .OBJ file")(
        "out-filepath-soil", "filepath for produced .OUT file (soil)")(
        "obj-filepath-soil", "filepath for produced .OBJ file (soil)");

    po::positional_options_description pos_opts_desc;
    pos_opts_desc.add("grayscale-images-directory", 1)
        .add("binary-images-directory", 1)
        .add("out-filepath", 1)
        .add("obj-filepath", 1)
        .add("out-filepath-soil", 1)
        .add("obj-filepath-soil", 1);

    po::options_description cmdline_options;
    cmdline_options.add(generic).add(hidden);
    auto args = po::command_line_parser(argc, argv)
                    .options(cmdline_options)
                    .positional(pos_opts_desc)
                    .run();

    po::variables_map vm;
    po::store(args, vm);
    po::notify(vm);

    if (vm.count("version")) {
      cout << argv[0] << " " << VERSION << endl;
      return 0;
    }

    // Validate options
    // If help requested
    if (vm.count("help") ||
        // If any arguments are missing
        !(vm.count("grayscale-images-directory") &&
          vm.count("binary-images-directory") && vm.count("out-filepath") &&
          vm.count("obj-filepath"))) {
      cout << "usage: " << argv[0]
           << " [-h] [-v] [-V] GRAYSCALE_IMAGE_DIRECTORY "
              "BINARY_IMAGE_DIRECTORY OUT_FILEPATH OBJ_FILEPATH "
           << endl;
      cout << generic << endl;
      return 0;
    }

    // If soil should be removed, but no output files are provided
    if (soil_removal_flag &&
        !(vm.count("out-filepath-soil") && vm.count("obj-filepath-soil"))) {
      cout << "usage: " << argv[0]
           << " [-h] [-v] [-V] REMOVE_SOIL_FLAG GRAYSCALE_IMAGE_DIRECTORY "
              "SAMPLING BINARY_IMAGE_DIRECTORY OUT_FILEPATH OBJ_FILEPATH "
           << endl;
      cout << generic << endl;
      return 1;
    }

    // Map program options
    grayscale_images_directory = vm["grayscale-images-directory"].as<string>();
    binary_images_directory = vm["binary-images-directory"].as<string>();
    filepath_out = vm["out-filepath"].as<string>();
    filepath_obj = vm["obj-filepath"].as<string>();

    cout << "Soil removal flag:\t" << std::boolalpha << soil_removal_flag
         << endl;
    cout << "Grayscale images:\t" << grayscale_images_directory << endl;
    cout << "Sampling:\t\t" << to_string(sampling) << endl;
    cout << "Binary images:\t\t" << binary_images_directory << endl;
    cout << "OUT filepath:\t\t" << filepath_out << endl;
    cout << "OBJ filepath:\t\t" << filepath_obj << endl;
    cout << "Manual flag:\t\t" << std::boolalpha << manual_segmentation << endl;
    cout << "Lower bound:\t\t" << to_string(lower_bound) << endl;
    cout << "Upper bound:\t\t" << to_string(upper_bound) << endl;

    // Perform segmentation
    if (soil_removal_flag) {
      filepath_out_soil = vm["out-filepath-soil"].as<string>();
      filepath_obj_soil = vm["obj-filepath-soil"].as<string>();
      cout << "OUT filepath (soil):\t" << filepath_out_soil << endl;
      cout << "OBJ filepath (soil):\t" << filepath_obj_soil << endl;

      segment(grayscale_images_directory, sampling, binary_images_directory,
              filepath_out, filepath_obj, filepath_out_soil, filepath_obj_soil);
    } else if (manual_segmentation) {
      user_defined_segment(grayscale_images_directory, sampling,
                           binary_images_directory, filepath_out, filepath_obj,
                           lower_bound, upper_bound);
    } else {
      segment(grayscale_images_directory, sampling, binary_images_directory,
              filepath_out, filepath_obj);
    }
    cout << "Finished processing " << grayscale_images_directory << ". Exiting."
         << endl;
  } catch (exception &e) {
    cerr << e.what() << endl;
  }
  return 0;
}
