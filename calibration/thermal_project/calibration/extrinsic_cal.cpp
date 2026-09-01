#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <CalibrationContracts.h>
#include <HeldOutVerifier.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <string>

/// \file depthimage.cpp
/// \brief Program that streams thermal images and saves it to thermal_images directory.

/// \brief Function to describe how to use the command line arguments
/// \param cmd Argument of the command line, here it is the program
void printUsage(char *cmd)
{
	char *cmdname = basename(cmd);
	printf(" Usage: %s [OPTION]...\n"
		   " -h             display this help and exit.\n"
		   " -r x		number of rows in the patterns (default 4).\n"
		   " -c x		number of columns in the patterns (default 5).\n"
           " -n x		number of image pairs (NEEDED).\n"
		   " Output:	Camera extrinsics between the thermal and rgb\n"
		   "                camera saved to an extrinsic.xml file.\n"
		   "", cmdname);
	return;
}

/// \brief Finds the pattern between thermal and color images to determine the extrinsics.
/// \param argc Number of command-line arguments.
/// \param argv Array of command-line arguments.
/// \param r Number of rows in the patterns.
/// \param c Number of columns in the patterns.
/// \param n Number of images pairs.
/// \return 0 if successful, -1 if failure.
int main(int argc, char **argv)
{
    int row = 4;
    int column = 5;
    int n = -1;
    for(int i=1; i < argc; i++)
	{
		if (std::strcmp(argv[i], "-h") == 0)
		{
			printUsage(argv[0]);
			exit(0);
		}
		else if (std::strcmp(argv[i], "-r") == 0)
		{
			if (i + 1 != argc)
			{
				int temp = std::stoi(argv[++i]);
				if (temp < 0 || temp > 255){
					std::cerr << "Error: Enter a valid number of rows." << std::endl;
					exit(1);
				}
				row = temp;
			}
            else
            {
				std::cerr << "Error: Enter the number of rows." << std::endl;
				exit(1);
			}
		}
		else if (std::strcmp(argv[i], "-c") == 0)
		{
			if (i + 1 != argc)
			{
				int temp = std::stoi(argv[++i]);
				if (temp < 0 || temp > 255){
					std::cerr << "Error: Enter a valid number of columns." << std::endl;
					exit(1);
				}
				column = temp;
			}
            else
            {
				std::cerr << "Error: Enter the number of columns." << std::endl;
				exit(1);
			}
		}
        else if (std::strcmp(argv[i], "-n") == 0)
		{
			if (i + 1 != argc)
			{
				int temp = std::stoi(argv[++i]);
				if (temp < 0 || temp > 255){
					std::cerr << "Error: Enter a valid number of images." << std::endl;
					exit(1);
				}
				n = temp;
			}
            else
            {
				std::cerr << "Error: Enter the number of columns." << std::endl;
				exit(1);
			}
		}
	}

    if (n == -1)
    {
        std::cerr << "Error: Enter the number of images. Check options with -h" << std::endl;
        exit(1);
    }
    std::string imageDirectory = IMAGE_DIRECTORY;
    std::string thermalimageDirectory = THERMAL_IMAGE_DIRECTORY;
    std::string calibrationfile = imageDirectory + "/../calibration.xml";
    cv::FileStorage fs(calibrationfile, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        std::cerr << "Failed to open calibration file: " << calibrationfile << std::endl;
        return -1;
    }
    cv::Mat cameraMatrixThermal, distCoeffsThermal;
    fs["cameraMatrix"] >> cameraMatrixThermal;
    fs["distCoeffs"] >> distCoeffsThermal;
    fs.release();
    if (!calibration::valid_calibration_values(cameraMatrixThermal, distCoeffsThermal))
    {
        std::cerr << "Thermal intrinsics are empty or non-finite." << std::endl;
        return -1;
    }

    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_RGB8, 15);
    pipe.start(cfg);
    auto color_stream = pipe.get_active_profile().get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
    if (color_stream.width() != 1280 || color_stream.height() != 720 ||
        color_stream.format() != RS2_FORMAT_RGB8 || color_stream.fps() != 15)
    {
        std::cerr << "Active color profile must be 1280x720@15 RGB8." << std::endl;
        return -1;
    }
    rs2_intrinsics intrinsicsColor = color_stream.get_intrinsics();
    cv::Mat cameraMatrixRealSense = (cv::Mat_<double>(3, 3) <<
        intrinsicsColor.fx, 0, intrinsicsColor.ppx,
        0, intrinsicsColor.fy, intrinsicsColor.ppy,
        0, 0, 1);

    cv::Mat distCoeffsRealSense = cv::Mat::zeros(5, 1, CV_64F);
    // Same coeff-aware rule as heldout_verify (single source of truth): accept
    // NONE, BROWN_CONRADY, or all-zero INVERSE_BROWN_CONRADY; reject non-zero
    // inverse coeffs (they must not be reinterpreted as OpenCV Brown-Conrady).
    if (!calibration::supported_realsense_color_distortion(intrinsicsColor))
    {
        std::cerr << "RealSense color distortion model "
                  << rs2_distortion_to_string(intrinsicsColor.model)
                  << " with non-zero coefficients is incompatible with OpenCV"
                     " Brown-Conrady coefficients." << std::endl;
        return -1;
    }
    if (intrinsicsColor.model == RS2_DISTORTION_BROWN_CONRADY)
    {
        for (int i = 0; i < 5; ++i)
        {
            distCoeffsRealSense.at<double>(i, 0) = intrinsicsColor.coeffs[i];
        }
    }
    if (!calibration::valid_calibration_values(cameraMatrixRealSense, distCoeffsRealSense))
    {
        std::cerr << "RealSense intrinsics are empty or non-finite." << std::endl;
        return -1;
    }

    // Set up chessboard parameters
    calibration::BoardContract board(row, column, 0.03);
    cv::Size patternSize = calibration::inner_corner_size(board);
    std::vector<cv::Point3f> objectPoints = calibration::object_points(board);
    std::vector<std::vector<cv::Point2f>> cornersColorVec, cornersThermalVec;
    std::vector<std::vector<cv::Point3f>> objectPointsVec;

    for (int imgIndex = 1; imgIndex <= n; ++imgIndex)
    {
        calibration::CapturePair pair;
        std::string pairError;
        if (!calibration::load_capture_pair(
                imageDirectory, thermalimageDirectory, imgIndex, pair, pairError))
        {
            std::cerr << pairError << std::endl;
            continue;
        }

        std::vector<cv::Point2f> cornersColor, cornersThermal;
        // findChessboardCornersSB (subpixel-native via CALIB_CB_ACCURACY) detects
        // the cut-out board where classic findChessboardCorners fails on RGB.
        bool foundColor = cv::findChessboardCornersSB(
            pair.color_gray, patternSize, cornersColor,
            cv::CALIB_CB_EXHAUSTIVE | cv::CALIB_CB_ACCURACY);
        bool foundThermal = cv::findChessboardCornersSB(
            pair.thermal_gray, patternSize, cornersThermal,
            cv::CALIB_CB_EXHAUSTIVE | cv::CALIB_CB_ACCURACY);
        if (!foundColor || !foundThermal)
        {
            std::cerr << "Chessboard corners not found in image pair " << imgIndex << std::endl;
            continue;
        }

        // Align the color and thermal corner sequences to one geometric origin so
        // corners[i] is the same physical corner in both (required by stereoCalibrate).
        calibration::canonicalize_corner_order(cornersColor);
        calibration::canonicalize_corner_order(cornersThermal);
        cornersColorVec.push_back(cornersColor);
        cornersThermalVec.push_back(cornersThermal);
        objectPointsVec.push_back(objectPoints);
    }

    if (!calibration::has_minimum_samples(cornersColorVec.size()) ||
        !calibration::has_minimum_samples(cornersThermalVec.size()))
    {
        std::cerr << "At least 10 valid paired chessboard detections are required." << std::endl;
        return -1;
    }

    cv::Mat R, T, E, F;
    double reprojectionError = cv::stereoCalibrate(objectPointsVec, cornersThermalVec, cornersColorVec,
                        cameraMatrixThermal, distCoeffsThermal,
                        cameraMatrixRealSense, distCoeffsRealSense,
                        cv::Size(), R, T, E, F,
                        cv::CALIB_FIX_INTRINSIC);

    if (!std::isfinite(reprojectionError) ||
        !calibration::valid_calibration_values(R, T) ||
        !calibration::valid_calibration_values(E, F))
    {
        std::cerr << "Stereo calibration returned empty or non-finite values." << std::endl;
        return -1;
    }

    std::cout << "Rotation Matrix:\n" << R << std::endl;
    std::cout << "Translation Vector:\n" << T << std::endl;

    std::string extfile = imageDirectory + "/../extrinsic.xml";
    cv::FileStorage fsOut(extfile, cv::FileStorage::WRITE);
    if (!fsOut.isOpened())
    {
        std::cerr << "Failed to open extrinsic.xml for writing." << std::endl;
        return -1;
    }
    fsOut << "R" << R;
    fsOut << "T" << T;
    fsOut << "E" << E;
    fsOut << "F" << F;
    fsOut.release();
    std::cout << "Extrinsic parameters and stereo calibration outputs saved to extrinsic.xml" << std::endl;
    std::cout << "Reprojection Error: " << reprojectionError << std::endl;

    return 0;
}
