#include "CalibrationContracts.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace calibration {
namespace {

const cv::Size kThermalImageSize(160, 120);
const cv::Size kColorImageSize(1280, 720);

std::string join_path(const std::string &directory, const std::string &basename)
{
    if (!directory.empty() && directory.back() == '/')
    {
        return directory + basename;
    }
    return directory + '/' + basename;
}

bool finite_matrix(const cv::Mat &matrix)
{
    return !matrix.empty() && cv::checkRange(matrix, true, nullptr);
}

bool valid_board(const BoardContract &board)
{
    return board.square_rows >= 2 && board.square_columns >= 2 &&
           std::isfinite(board.square_size_m) && board.square_size_m > 0.0;
}

}  // namespace

void canonicalize_corner_order(std::vector<cv::Point2f> &corners)
{
    // Force a fixed GEOMETRIC origin (top-left-most corner first), independent of
    // image contrast polarity. findChessboardCornersSB returns the 12 corners in
    // grid-scan order, so corners.front()/back() are opposite diagonal corners; a
    // full reversal is the board's only ordering ambiguity for a 4x3 (non-square)
    // grid. Applying the SAME geometric rule to both the color and thermal corner
    // vectors of a co-mounted, same-side pair makes index i refer to the same
    // physical corner in both, which stereoCalibrate / solvePnP require. Do NOT
    // apply to single-camera intrinsics: there each view pairs with its own
    // object-point template, so a 180-degree relabel is just a valid alternate pose.
    if (corners.size() < 2)
    {
        return;
    }
    const cv::Point2f &first = corners.front();
    const cv::Point2f &last = corners.back();
    if (last.y < first.y || (last.y == first.y && last.x < first.x))
    {
        std::reverse(corners.begin(), corners.end());
    }
}

cv::Size inner_corner_size(const BoardContract &board)
{
    if (!valid_board(board))
    {
        throw std::invalid_argument("Physical checkerboard dimensions must be at least 2x2");
    }
    return cv::Size(board.square_columns - 1, board.square_rows - 1);
}

std::vector<cv::Point3f> object_points(const BoardContract &board)
{
    const cv::Size inner = inner_corner_size(board);
    std::vector<cv::Point3f> points;
    points.reserve(static_cast<std::size_t>(inner.area()));
    for (int row = 0; row < inner.height; ++row)
    {
        for (int column = 0; column < inner.width; ++column)
        {
            points.emplace_back(
                static_cast<float>(column * board.square_size_m),
                static_cast<float>(row * board.square_size_m),
                0.0F);
        }
    }
    return points;
}

bool has_minimum_samples(std::size_t count)
{
    return count >= kMinimumSamples;
}

bool valid_calibration_values(const cv::Mat &first, const cv::Mat &second)
{
    return finite_matrix(first) && finite_matrix(second);
}

bool load_capture_pair(
    const std::string &color_directory,
    const std::string &thermal_directory,
    int index,
    CapturePair &pair,
    std::string &error)
{
    const std::string color_path = join_path(
        color_directory, "color_image_" + std::to_string(index) + ".png");
    const std::string thermal_path = join_path(
        thermal_directory, "thermal_grayimage_" + std::to_string(index) + ".png");
    const cv::Mat color = cv::imread(color_path, cv::IMREAD_COLOR);
    pair.thermal_gray = cv::imread(thermal_path, cv::IMREAD_GRAYSCALE);
    if (color.empty() || pair.thermal_gray.empty())
    {
        error = "Failed to load capture pair " + std::to_string(index) +
                " (expected " + color_path + " and " + thermal_path + ')';
        return false;
    }
    if (color.size() != kColorImageSize)
    {
        error = "Color image " + color_path + " must be 1280x720";
        return false;
    }
    if (pair.thermal_gray.size() != kThermalImageSize)
    {
        error = "Thermal image " + thermal_path + " must be 160x120";
        return false;
    }
    cv::cvtColor(color, pair.color_gray, cv::COLOR_BGR2GRAY);
    return true;
}

}  // namespace calibration
