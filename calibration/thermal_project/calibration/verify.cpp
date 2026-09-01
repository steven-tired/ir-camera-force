#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

/// \file verify.cpp
/// \brief Program that checks the calibration by undistorting the thermal image

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

/// \brief Verifies camera calibration by undistorting images.
/// \param argc Number of command-line arguments.
/// \param argv Array of command-line arguments.
/// \param n Number of images.
/// \return 0 if successful, -1 if failure.
int main(int argc, char **argv)
{
    int n = -1;
    for(int i=1; i < argc; i++)
	{
		if (strcmp(argv[i], "-h") == 0)
		{
			printUsage(argv[0]);
			exit(0);
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
	}

    if (n == -1)
    {
        cerr << "Error: Enter the number of images. Check options with -h" << endl;
        exit(1);
    }

    Mat cameraMatrix, distCoeffs;
    FileStorage fs("../calibration.xml", FileStorage::READ); // Add the path to your calibration file
    if (!fs.isOpened())
    {
        cout << "Failed to open calibration file" << endl;
        return -1;
    }
    fs["cameraMatrix"] >> cameraMatrix;
    fs["distCoeffs"] >> distCoeffs;
    fs.release();

    string imageDirectory = THERMAL_IMAGE_DIRECTORY;
    vector<string> images;
    for (int i = 1; i <= n; i++)
    {
        string imagePath = imageDirectory + "/thermal_grayimage_" + to_string(i) + ".png";
        images.push_back(imagePath);
    }

    for (size_t i = 0; i < images.size(); i++)
    {
        Mat image = imread(images[i], IMREAD_GRAYSCALE);
        if (image.empty())
        {
            cout << "Cannot open image file: " << images[i] << endl;
            continue;
        }
        Mat undistortedImage;
        undistort(image, undistortedImage, cameraMatrix, distCoeffs);
        namedWindow("Original Image", WINDOW_AUTOSIZE);
        imshow("Original Image", image);
        namedWindow("Undistorted Image", WINDOW_AUTOSIZE);
        imshow("Undistorted Image", undistortedImage);
        waitKey(0);
    }
    destroyAllWindows();

    return 0;
}
