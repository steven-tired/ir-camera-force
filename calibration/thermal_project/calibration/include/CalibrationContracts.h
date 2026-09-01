#pragma once

#include <opencv2/core.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace calibration {

constexpr std::size_t kMinimumSamples = 10;

struct BoardContract
{
    BoardContract(int rows = 4, int columns = 5, double size_m = 0.03)
        : square_rows(rows), square_columns(columns), square_size_m(size_m)
    {
    }

    int square_rows;
    int square_columns;
    double square_size_m;
};

struct CapturePair
{
    cv::Mat color_gray;
    cv::Mat thermal_gray;
};

// Normalize a detected chessboard corner vector to a fixed geometric top-left
// origin so paired color/thermal detections index the same physical corner.
// Apply to both members of a stereo pair; never to single-camera intrinsics.
void canonicalize_corner_order(std::vector<cv::Point2f> &corners);

cv::Size inner_corner_size(const BoardContract &board);
std::vector<cv::Point3f> object_points(const BoardContract &board);
bool has_minimum_samples(std::size_t count);
bool valid_calibration_values(const cv::Mat &first, const cv::Mat &second);

bool load_capture_pair(
    const std::string &color_directory,
    const std::string &thermal_directory,
    int index,
    CapturePair &pair,
    std::string &error);

}  // namespace calibration
