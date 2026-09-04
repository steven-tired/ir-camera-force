#pragma once

#include <opencv2/core.hpp>
#include <librealsense2/h/rs_types.h>

#include <cstddef>
#include <string>
#include <vector>

namespace calibration {

struct ImageProjectionErrors
{
    int image_index = 0;
    std::vector<double> point_errors_px;
    double max_error_px = 0.0;
};

struct ProjectionSummary
{
    std::vector<ImageProjectionErrors> images;
    double global_max_error_px = 0.0;
    double rms_error_px = 0.0;
    std::size_t point_count = 0;
};

struct ValidationFailure
{
    int image_index;
    std::string reason;
};

struct HeldOutOptions
{
    bool help_requested = false;
    std::string color_directory;
    std::string thermal_directory;
    std::string intrinsic_xml;
    std::string extrinsic_xml;
    std::string output_json;
};

std::vector<cv::Point3f> color_points_to_thermal(
    const std::vector<cv::Point3f> &color_points,
    const cv::Mat &rotation_thermal_to_color,
    const cv::Mat &translation_thermal_to_color);

ProjectionSummary summarize_projection_errors(
    const std::vector<int> &image_indices,
    const std::vector<std::vector<cv::Point2f>> &projected,
    const std::vector<std::vector<cv::Point2f>> &detected);

constexpr double kHeldOutThresholdPx = 3.0;
constexpr int kHeldOutFirstIndex = 25;
constexpr int kHeldOutImageCount = 12;
constexpr int kCornersPerImage = 12;

bool passes_projection_gate(const ProjectionSummary &summary);

bool write_projection_report_json(
    const std::string &path,
    const ProjectionSummary &summary,
    const std::vector<ValidationFailure> &failures,
    std::string &error);

bool parse_heldout_options(
    int argc, char **argv, HeldOutOptions &options, std::string &error);

// Accept a RealSense color distortion configuration that maps to OpenCV's
// plumb-bob with the coefficients we actually use: NONE (zero distortion),
// BROWN_CONRADY (coeffs copied as-is), or INVERSE_BROWN_CONRADY ONLY when all
// coefficients are zero (then it is identical to zero distortion). Non-zero
// inverse coefficients are rejected: they must not be reinterpreted as
// OpenCV Brown-Conrady. Coeff-aware so both extrinsic and heldout share one rule.
bool supported_realsense_color_distortion(const rs2_intrinsics &intrinsics);

}  // namespace calibration
