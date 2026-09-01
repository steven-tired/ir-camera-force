#include "CalibrationContracts.h"
#include "HeldOutVerifier.h"

#include <librealsense2/rs.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr char kExpectedRealSenseSerial[] = "233522078685";

void print_usage(const char *program)
{
    std::cout << "Usage: " << program << " --color-dir PATH --thermal-dir PATH "
              << "--intrinsic PATH --extrinsic PATH --output PATH\n";
}

bool load_matrix_file(
    const std::string &path,
    const char *first_key,
    const char *second_key,
    cv::Mat &first,
    cv::Mat &second,
    std::string &error)
{
    try
    {
        cv::FileStorage storage(path, cv::FileStorage::READ);
        if (!storage.isOpened())
        {
            error = "Failed to open XML: " + path;
            return false;
        }
        storage[first_key] >> first;
        storage[second_key] >> second;
        if (!calibration::valid_calibration_values(first, second))
        {
            error = "XML has empty or non-finite matrices: " + path;
            return false;
        }
        return true;
    }
    catch (const cv::Exception &exception)
    {
        error = "Failed to parse XML " + path + ": " + exception.what();
        return false;
    }
}

bool load_extrinsic_file(
    const std::string &path,
    cv::Mat &rotation,
    cv::Mat &translation,
    cv::Mat &essential,
    cv::Mat &fundamental,
    std::string &error)
{
    try
    {
        cv::FileStorage storage(path, cv::FileStorage::READ);
        if (!storage.isOpened())
        {
            error = "Failed to open XML: " + path;
            return false;
        }
        storage["R"] >> rotation;
        storage["T"] >> translation;
        storage["E"] >> essential;
        storage["F"] >> fundamental;
        if (!calibration::valid_calibration_values(rotation, translation) ||
            !calibration::valid_calibration_values(essential, fundamental))
        {
            error = "Extrinsic XML must contain finite R, T, E and F matrices: " + path;
            return false;
        }
        return true;
    }
    catch (const cv::Exception &exception)
    {
        error = "Failed to parse XML " + path + ": " + exception.what();
        return false;
    }
}

bool finite_points(const std::vector<cv::Point3f> &points)
{
    for (const auto &point : points)
    {
        if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
            !std::isfinite(point.z))
        {
            return false;
        }
    }
    return true;
}

bool valid_distortion_shape(const cv::Mat &distortion)
{
    if (distortion.rows != 1 && distortion.cols != 1)
    {
        return false;
    }
    const std::size_t count = distortion.total();
    return count == 4 || count == 5 || count == 8 || count == 12 || count == 14;
}

}  // namespace

int main(int argc, char **argv)
{
    calibration::HeldOutOptions options;
    std::string error;
    if (!calibration::parse_heldout_options(argc, argv, options, error))
    {
        std::cerr << error << '\n';
        print_usage(argv[0]);
        return 1;
    }
    if (options.help_requested)
    {
        print_usage(argv[0]);
        return 0;
    }

    cv::Mat thermal_camera_matrix;
    cv::Mat thermal_distortion;
    cv::Mat rotation_thermal_to_color;
    cv::Mat translation_thermal_to_color;
    cv::Mat essential_matrix;
    cv::Mat fundamental_matrix;
    if (!load_matrix_file(
            options.intrinsic_xml,
            "cameraMatrix",
            "distCoeffs",
            thermal_camera_matrix,
            thermal_distortion,
            error) ||
        !load_extrinsic_file(
            options.extrinsic_xml,
            rotation_thermal_to_color,
            translation_thermal_to_color,
            essential_matrix,
            fundamental_matrix,
            error))
    {
        std::cerr << error << '\n';
        return 1;
    }
    if (thermal_camera_matrix.rows != 3 || thermal_camera_matrix.cols != 3 ||
        !valid_distortion_shape(thermal_distortion) ||
        rotation_thermal_to_color.rows != 3 ||
        rotation_thermal_to_color.cols != 3 ||
        translation_thermal_to_color.rows != 3 ||
        translation_thermal_to_color.cols != 1 ||
        essential_matrix.rows != 3 || essential_matrix.cols != 3 ||
        fundamental_matrix.rows != 3 || fundamental_matrix.cols != 3)
    {
        std::cerr << "Calibration XML contains an invalid matrix shape.\n";
        return 1;
    }

    cv::Mat color_camera_matrix;
    cv::Mat color_distortion = cv::Mat::zeros(5, 1, CV_64F);
    try
    {
        rs2::pipeline pipeline;
        rs2::config config;
        config.enable_device(kExpectedRealSenseSerial);
        config.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_RGB8, 15);
        const rs2::pipeline_profile active = pipeline.start(config);
        const std::string serial = active.get_device().get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
        if (serial != kExpectedRealSenseSerial)
        {
            std::cerr << "Active RealSense serial does not match the compiled contract.\n";
            return 1;
        }
        const auto color_stream = active.get_stream(RS2_STREAM_COLOR)
            .as<rs2::video_stream_profile>();
        if (color_stream.width() != 1280 || color_stream.height() != 720 ||
            color_stream.format() != RS2_FORMAT_RGB8 || color_stream.fps() != 15)
        {
            std::cerr << "Active color profile must be 1280x720 RGB8@15.\n";
            return 1;
        }
        const rs2_intrinsics intrinsics = color_stream.get_intrinsics();
        if (!calibration::supported_realsense_color_distortion(intrinsics))
        {
            std::cerr << "Unsupported RealSense color distortion: model "
                      << rs2_distortion_to_string(intrinsics.model)
                      << " with non-zero coefficients cannot be used as OpenCV"
                         " Brown-Conrady.\n";
            return 1;
        }
        color_camera_matrix = (cv::Mat_<double>(3, 3) <<
            intrinsics.fx, 0.0, intrinsics.ppx,
            0.0, intrinsics.fy, intrinsics.ppy,
            0.0, 0.0, 1.0);
        if (intrinsics.model == RS2_DISTORTION_BROWN_CONRADY)
        {
            for (int coefficient = 0; coefficient < 5; ++coefficient)
            {
                color_distortion.at<double>(coefficient, 0) =
                    intrinsics.coeffs[coefficient];
            }
        }
        pipeline.stop();
    }
    catch (const rs2::error &exception)
    {
        std::cerr << "RealSense precondition failed: " << exception.what() << '\n';
        return 1;
    }
    if (!calibration::valid_calibration_values(
            color_camera_matrix, color_distortion))
    {
        std::cerr << "RealSense intrinsics are empty or non-finite.\n";
        return 1;
    }

    const calibration::BoardContract board(4, 5, 0.03);
    const cv::Size pattern_size = calibration::inner_corner_size(board);
    const std::vector<cv::Point3f> board_points = calibration::object_points(board);
    std::vector<int> evaluated_indices;
    std::vector<std::vector<cv::Point2f>> projected_images;
    std::vector<std::vector<cv::Point2f>> detected_images;
    std::vector<calibration::ValidationFailure> validation_failures;

    for (int index = calibration::kHeldOutFirstIndex;
         index < calibration::kHeldOutFirstIndex + calibration::kHeldOutImageCount;
         ++index)
    {
        calibration::CapturePair pair;
        std::string pair_error;
        if (!calibration::load_capture_pair(
                options.color_directory,
                options.thermal_directory,
                index,
                pair,
                pair_error))
        {
            validation_failures.push_back({index, pair_error});
            continue;
        }

        try
        {
            std::vector<cv::Point2f> color_corners;
            std::vector<cv::Point2f> thermal_corners;
            // findChessboardCornersSB (subpixel-native) matches the detector used
            // by camera_calibration/extrinsic; classic fails on the cut-out board.
            const bool color_found = cv::findChessboardCornersSB(
                pair.color_gray, pattern_size, color_corners,
                cv::CALIB_CB_EXHAUSTIVE | cv::CALIB_CB_ACCURACY);
            const bool thermal_found = cv::findChessboardCornersSB(
                pair.thermal_gray, pattern_size, thermal_corners,
                cv::CALIB_CB_EXHAUSTIVE | cv::CALIB_CB_ACCURACY);
            if (!color_found || !thermal_found ||
                color_corners.size() != calibration::kCornersPerImage ||
                thermal_corners.size() != calibration::kCornersPerImage)
            {
                validation_failures.push_back({index, "complete 4x3 corners not detected"});
                continue;
            }
            // Same geometric-origin alignment as extrinsic_cal, so color_corners[i]
            // and thermal_corners[i] are the same physical corner.
            calibration::canonicalize_corner_order(color_corners);
            calibration::canonicalize_corner_order(thermal_corners);

            cv::Mat rotation_vector;
            cv::Mat translation_board_to_color;
            if (!cv::solvePnP(
                    board_points,
                    color_corners,
                    color_camera_matrix,
                    color_distortion,
                    rotation_vector,
                    translation_board_to_color) ||
                !calibration::valid_calibration_values(
                    rotation_vector, translation_board_to_color))
            {
                validation_failures.push_back({index, "color solvePnP failed"});
                continue;
            }

            cv::Mat rotation_board_to_color;
            cv::Rodrigues(rotation_vector, rotation_board_to_color);
            std::vector<cv::Point3f> color_points;
            color_points.reserve(board_points.size());
            for (const cv::Point3f &board_point : board_points)
            {
                const cv::Mat x_board = (cv::Mat_<double>(3, 1) <<
                    board_point.x, board_point.y, board_point.z);
                const cv::Mat x_color =
                    rotation_board_to_color * x_board + translation_board_to_color;
                color_points.emplace_back(
                    static_cast<float>(x_color.at<double>(0, 0)),
                    static_cast<float>(x_color.at<double>(1, 0)),
                    static_cast<float>(x_color.at<double>(2, 0)));
            }
            if (!finite_points(color_points))
            {
                validation_failures.push_back({index, "non-finite color-space point"});
                continue;
            }

            const std::vector<cv::Point3f> thermal_points =
                calibration::color_points_to_thermal(
                    color_points,
                    rotation_thermal_to_color,
                    translation_thermal_to_color);
            bool positive_depth = true;
            for (const cv::Point3f &point : thermal_points)
            {
                positive_depth = positive_depth && point.z > 0.0F;
            }
            if (!positive_depth)
            {
                validation_failures.push_back({index, "thermal projection depth is not positive"});
                continue;
            }

            std::vector<cv::Point2f> projected;
            cv::projectPoints(
                thermal_points,
                cv::Vec3d(0.0, 0.0, 0.0),
                cv::Vec3d(0.0, 0.0, 0.0),
                thermal_camera_matrix,
                thermal_distortion,
                projected);
            if (projected.size() != calibration::kCornersPerImage)
            {
                validation_failures.push_back({index, "thermal projection count mismatch"});
                continue;
            }
            evaluated_indices.push_back(index);
            projected_images.push_back(projected);
            detected_images.push_back(thermal_corners);
        }
        catch (const std::exception &exception)
        {
            validation_failures.push_back({index, exception.what()});
        }
    }

    calibration::ProjectionSummary summary;
    if (!evaluated_indices.empty())
    {
        try
        {
            summary = calibration::summarize_projection_errors(
                evaluated_indices, projected_images, detected_images);
        }
        catch (const std::exception &exception)
        {
            validation_failures.push_back({0, exception.what()});
        }
    }
    if (!calibration::write_projection_report_json(
            options.output_json, summary, validation_failures, error))
    {
        std::cerr << error << '\n';
        return 1;
    }
    return validation_failures.empty() &&
           calibration::passes_projection_gate(summary) ? 0 : 2;
}
