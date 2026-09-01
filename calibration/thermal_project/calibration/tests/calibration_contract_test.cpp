#include "CalibrationContracts.h"

#include <opencv2/imgcodecs.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/stat.h>

namespace {

int failures = 0;

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            std::cerr << __FILE__ << ':' << __LINE__                            \
                      << ": CHECK failed: " #condition << '\n';                \
            ++failures;                                                         \
        }                                                                       \
    } while (false)

std::string make_temp_directory()
{
    char path[] = "/tmp/thermal-calibration-contract-XXXXXX";
    char *created = mkdtemp(path);
    if (created == nullptr)
    {
        throw std::runtime_error("mkdtemp failed");
    }
    return created;
}

void test_board_contract()
{
    calibration::BoardContract board{4, 5, 0.03};
    const cv::Size inner = calibration::inner_corner_size(board);
    CHECK(inner == cv::Size(4, 3));

    const std::vector<cv::Point3f> points = calibration::object_points(board);
    CHECK(points.size() == 12);
    CHECK(std::abs(points.at(1).x - 0.03F) < 1e-6F);
    CHECK(std::abs(points.at(4).y - 0.03F) < 1e-6F);
}

void test_capture_pair_loading_and_grayscale_conversion()
{
    const std::string root = make_temp_directory();
    const std::string color_dir = root + "/color";
    const std::string thermal_dir = root + "/thermal";
    CHECK(mkdir(color_dir.c_str(), 0700) == 0);
    CHECK(mkdir(thermal_dir.c_str(), 0700) == 0);

    const cv::Mat color(720, 1280, CV_8UC3, cv::Scalar(10, 20, 30));
    const cv::Mat thermal(120, 160, CV_8UC1, cv::Scalar(40));
    CHECK(cv::imwrite(color_dir + "/color_image_7.png", color));
    CHECK(cv::imwrite(thermal_dir + "/thermal_grayimage_7.png", thermal));

    calibration::CapturePair pair;
    std::string error;
    CHECK(calibration::load_capture_pair(color_dir, thermal_dir, 7, pair, error));
    CHECK(pair.color_gray.channels() == 1);
    CHECK(pair.thermal_gray.channels() == 1);
    CHECK(pair.color_gray.size() == cv::Size(1280, 720));
    CHECK(pair.thermal_gray.size() == cv::Size(160, 120));
}

void test_capture_pair_size_mismatch_is_rejected()
{
    const std::string root = make_temp_directory();
    const std::string color_dir = root + "/color";
    const std::string thermal_dir = root + "/thermal";
    CHECK(mkdir(color_dir.c_str(), 0700) == 0);
    CHECK(mkdir(thermal_dir.c_str(), 0700) == 0);

    CHECK(cv::imwrite(color_dir + "/color_image_1.png",
                      cv::Mat(480, 640, CV_8UC3, cv::Scalar(0, 0, 0))));
    CHECK(cv::imwrite(thermal_dir + "/thermal_grayimage_1.png",
                      cv::Mat(120, 160, CV_8UC1, cv::Scalar(0))));

    calibration::CapturePair pair;
    std::string error;
    CHECK(!calibration::load_capture_pair(color_dir, thermal_dir, 1, pair, error));
    CHECK(error.find("1280x720") != std::string::npos);

    CHECK(cv::imwrite(color_dir + "/color_image_2.png",
                      cv::Mat(720, 1280, CV_8UC3, cv::Scalar(0, 0, 0))));
    CHECK(cv::imwrite(thermal_dir + "/thermal_grayimage_2.png",
                      cv::Mat(128, 160, CV_8UC1, cv::Scalar(0))));
    error.clear();
    CHECK(!calibration::load_capture_pair(color_dir, thermal_dir, 2, pair, error));
    CHECK(error.find("160x120") != std::string::npos);
}

void test_minimum_sample_count()
{
    CHECK(!calibration::has_minimum_samples(9));
    CHECK(calibration::has_minimum_samples(10));
}

void test_invalid_calibration_values_are_rejected()
{
    const cv::Mat valid = cv::Mat::eye(3, 3, CV_64F);
    const cv::Mat distortion = cv::Mat::zeros(1, 5, CV_64F);
    CHECK(calibration::valid_calibration_values(valid, distortion));
    CHECK(!calibration::valid_calibration_values(cv::Mat(), distortion));
    CHECK(!calibration::valid_calibration_values(valid, cv::Mat()));

    cv::Mat non_finite = valid.clone();
    non_finite.at<double>(0, 0) = std::numeric_limits<double>::quiet_NaN();
    CHECK(!calibration::valid_calibration_values(non_finite, distortion));
}

void test_canonicalize_corner_order()
{
    // A 4x3 scan-order corner vector (top-left origin, row-major) must be left as-is.
    std::vector<cv::Point2f> forward;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c)
            forward.emplace_back(static_cast<float>(c), static_cast<float>(r));
    std::vector<cv::Point2f> forward_copy = forward;
    calibration::canonicalize_corner_order(forward_copy);
    CHECK(forward_copy.front() == forward.front());
    CHECK(forward_copy.back() == forward.back());

    // The 180-degree reversal (bottom-right origin) must be flipped back to match.
    std::vector<cv::Point2f> reversed(forward.rbegin(), forward.rend());
    calibration::canonicalize_corner_order(reversed);
    CHECK(reversed.size() == forward.size());
    bool identical = true;
    for (std::size_t i = 0; i < forward.size(); ++i)
        identical = identical && (reversed.at(i) == forward.at(i));
    CHECK(identical);

    // Idempotent: canonicalizing an already-canonical vector changes nothing.
    std::vector<cv::Point2f> again = reversed;
    calibration::canonicalize_corner_order(again);
    CHECK(again.front() == forward.front());
    CHECK(again.back() == forward.back());
}

}  // namespace

int main()
{
    test_board_contract();
    test_capture_pair_loading_and_grayscale_conversion();
    test_capture_pair_size_mismatch_is_rejected();
    test_minimum_sample_count();
    test_invalid_calibration_values_are_rejected();
    test_canonicalize_corner_order();

    if (failures != 0)
    {
        std::cerr << failures << " calibration contract checks failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "All calibration contract checks passed\n";
    return EXIT_SUCCESS;
}
