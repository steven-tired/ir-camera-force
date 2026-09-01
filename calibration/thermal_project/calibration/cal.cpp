#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <CalibrationContracts.h>
#include <cmath>
#include <iostream>
#include <vector>

/// \file cal.cpp
/// \brief Program that calibrates the thermal camera

using namespace cv;
using namespace std;

/// \brief Function to describe how to use the command line arguments
/// \param cmd Argument of the command line, here it is the program
void printUsage(char *cmd)
{
	char *cmdname = basename(cmd);
	printf(" Usage: %s [OPTION]...\n"
		   " -h             display this help and exit.\n"
		   " -r x		number of rows in the pattern (default 4).\n"
		   " -c x		number of columns in the pattern (default 5).\n"
           " -n x		number of images (NEEDED).\n"
		   " -pat x		calibration patterns.\n"
		   "		1: chessboard (default)\n"
		   "		2: symmetric dots (default)\n"
		   " Output:	Camera intrinsics to a calibration.xml file.\n"
		   "", cmdname);
	return;
}

/// \brief Finds the circular dots in a symmetric dot pattern.
/// \param image Thermal Image that contains the pattern.
/// \param patternSize Pattern size in rows and columns.
/// \param centers Centers of the dots which will be calculated.
/// \return 0 if failure, non-zero if successful.
bool findDotPattern(const Mat& image, Size patternSize, vector<Point2f>& centers)
{
    // Change according to your dot pattern
    SimpleBlobDetector::Params params;
    // params.minThreshold = 127;
    // params.maxThreshold = 255;
    params.filterByArea = true;
    params.minArea = 2;
    params.maxArea = 10000;
    params.filterByCircularity = true;
    params.minCircularity = 0.5;
    // params.filterByConvexity = true;
    // params.minConvexity = 0.7;
    // params.filterByInertia = true;
    // params.minInertiaRatio = 0.1;

    Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
    Mat thresholded;
    adaptiveThreshold(image, thresholded, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 13, 2);
    imshow("Thresholded Image", thresholded);
    waitKey(0);

    return findCirclesGrid(thresholded, patternSize, centers, CALIB_CB_ASYMMETRIC_GRID, detector);
}

/// \brief Finds the pattern in the images for camera calibration.
/// \param argc Number of command-line arguments.
/// \param argv Array of command-line arguments.
/// \param r Number of rows in the pattern.
/// \param c Number of columns in the pattern.
/// \param n Number of images.
/// \param pat Pattern type.
/// \return 0 if successful, -1 if failure.
int main(int argc, char **argv)
{
    int row = 4;
    int column = 5;
    int pattern = 1;
    int n = -1;
    for(int i=1; i < argc; i++)
	{
		if (strcmp(argv[i], "-h") == 0)
		{
			printUsage(argv[0]);
			exit(0);
		}
		else if (strcmp(argv[i], "-r") == 0)
		{
			if (i + 1 != argc)
			{
				int temp = stoi(argv[++i]);
				if (temp < 0 || temp > 255){
					cerr << "Error: Enter a valid number of rows." << endl;
					exit(1);
				}
				row = temp;
			}
            else
            {
				cerr << "Error: Enter the number of rows." << endl;
				exit(1);
			}
		}
		else if (strcmp(argv[i], "-c") == 0)
		{
			if (i + 1 != argc)
			{
				int temp = stoi(argv[++i]);
				if (temp < 0 || temp > 255){
					cerr << "Error: Enter a valid number of columns." << endl;
					exit(1);
				}
				column = temp;
			}
            else
            {
				cerr << "Error: Enter the number of columns." << endl;
				exit(1);
			}
		}
        else if (strcmp(argv[i], "-n") == 0)
		{
			if (i + 1 != argc)
			{
				int temp = stoi(argv[++i]);
				if (temp < 0 || temp > 255){
					cerr << "Error: Enter a valid number of images." << endl;
					exit(1);
				}
				n = temp;
			}
            else
            {
				cerr << "Error: Enter the number of images." << endl;
				exit(1);
			}
		}
		else if (strcmp(argv[i], "-pat") == 0)
		{
			if (i + 1 != argc)
			{
				int temp = stoi(argv[++i]);
				if (temp != 1 && temp != 2){
					cerr << "Error: Enter a valid option." << endl;
					exit(1);
				}
				pattern = temp;
			}
            else
            {
				cerr << "Error: Enter an option. Check options with -h" << endl;
				exit(1);
			}
		}
	}

    if (n == -1)
    {
        cerr << "Error: Enter the number of images. Check options with -h" << endl;
        exit(1);
    }
    Size patternSize;
    if (pattern == 1)
    {
        patternSize = calibration::inner_corner_size(
            calibration::BoardContract(row, column, 0.03));
    }
    else
    {
        patternSize = Size(row, column);
    }
    vector<vector<Point2f>> imagePoints;
    vector<vector<Point3f>> objectPoints;

    vector<Point3f> obj;
    const float objectPointSpacing = pattern == 1 ? 0.03f : 1.0f;
    for (int i = 0; i < patternSize.height; i++)
    {
        for (int j = 0; j < patternSize.width; j++)
        {
            obj.push_back(Point3f(
                (float)j * objectPointSpacing,
                (float)i * objectPointSpacing,
                0));
        }
    }

    string imageDirectory = THERMAL_IMAGE_DIRECTORY;
    vector<string> images;
    for (int i = 1; i <= n; i++)
    {
        string imagePath = imageDirectory + "/thermal_grayimage_" + to_string(i) + ".png";
        images.push_back(imagePath);
    }

    Size imageSize;
    for (size_t i = 0; i < images.size(); i++)
    {
        Mat image = imread(images[i], IMREAD_GRAYSCALE);
        if (image.empty())
        {
            cout << "Cannot open image file: " << images[i] << endl;
            continue;
        }
        if (image.size() != Size(160, 120))
        {
            cerr << "Thermal image must be 160x120: " << images[i] << endl;
            continue;
        }
        if (imageSize.area() > 0 && image.size() != imageSize)
        {
            cerr << "Image size differs from the first accepted image: " << images[i] << endl;
            continue;
        }

        if (pattern == 1)
        {
            vector<Point2f> corners;
            // findChessboardCornersSB robustly detects the laser-cut cut-out board
            // (classic findChessboardCorners fails on it, esp. in RGB). SB is
            // subpixel-native via CALIB_CB_ACCURACY, so no cornerSubPix follows.
            bool patternFound = findChessboardCornersSB(image, patternSize, corners, CALIB_CB_EXHAUSTIVE | CALIB_CB_ACCURACY);

            if (patternFound)
            {
                if (imageSize.area() == 0)
                {
                    imageSize = image.size();
                }
                drawChessboardCorners(image, patternSize, Mat(corners), patternFound);
                imagePoints.push_back(corners);
                objectPoints.push_back(obj);
                imshow("Chessboard Pattern", image);
            }
            else
            {
                cout << "Pattern not found in image: " << images[i] << endl;
            }
        }

        if (pattern == 2)
        {
            Mat enhancedImage;
            equalizeHist(image, enhancedImage);
            imshow("Enhanced", enhancedImage);
            waitKey(0);

            vector<Point2f> centers;
            bool patternFound = findDotPattern(enhancedImage, patternSize, centers);

            if (patternFound)
            {
                if (imageSize.area() == 0)
                {
                    imageSize = image.size();
                }
                imagePoints.push_back(centers);
                objectPoints.push_back(obj);

                drawChessboardCorners(image, patternSize, Mat(centers), patternFound);
                imshow("Dot Pattern", image);
                waitKey(0);
            }
            else
            {
                cout << "Pattern not found in image: " << images[i] << endl;
            }
        }
    }
    destroyAllWindows();

    if (calibration::has_minimum_samples(imagePoints.size()) &&
        calibration::has_minimum_samples(objectPoints.size()))
    {
        // Camera calibration
        Mat cameraMatrix, distCoeffs;
        vector<Mat> rvecs, tvecs;
        double rms = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs);
        if (!std::isfinite(rms) ||
            !calibration::valid_calibration_values(cameraMatrix, distCoeffs))
        {
            cerr << "Error: Calibration returned empty or non-finite values." << endl;
            return -1;
        }
        cout << "Camera matrix: \n" << cameraMatrix << endl;
        cout << "Distortion coefficients: \n" << distCoeffs << endl;

        string calibrationfile = imageDirectory + "/../calibration.xml";
        FileStorage fs(calibrationfile, FileStorage::WRITE);
        if (!fs.isOpened())
        {
            cerr << "Error: Cannot open calibration.xml for writing." << endl;
            return -1;
        }
        fs << "cameraMatrix" << cameraMatrix;
        fs << "distCoeffs" << distCoeffs;
        fs.release();
    }
    else
    {
        cerr << "Error: At least 10 valid patterns are required for calibration." << endl;
        return -1;
    }

    return 0;
}
