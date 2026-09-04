#include "HeldOutVerifier.h"

#include <array>
#include <opencv2/core.hpp>

#include <cmath>
#include <cerrno>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

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

void check_invalid_argument(const std::function<void()> &call)
{
    try
    {
        call();
        CHECK(false);
    }
    catch (const std::invalid_argument &)
    {
    }
}

void test_transform_inverts_non_identity_thermal_to_color_pose()
{
    const cv::Mat rotation = (cv::Mat_<double>(3, 3) <<
        0.0, -1.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 0.0, 1.0);
    const cv::Mat translation = (cv::Mat_<double>(3, 1) << 1.0, 2.0, 3.0);
    const std::vector<cv::Point3f> color_points{
        cv::Point3f(1.0F, 3.0F, 4.0F),
        cv::Point3f(-1.0F, 2.0F, 2.0F),
    };

    const auto thermal = calibration::color_points_to_thermal(
        color_points, rotation, translation);
    CHECK(thermal.size() == 2);
    CHECK(cv::norm(thermal.at(0) - cv::Point3f(1.0F, 0.0F, 1.0F)) < 1e-6);
    CHECK(cv::norm(thermal.at(1) - cv::Point3f(0.0F, 2.0F, -1.0F)) < 1e-6);
}

void test_projection_summary_keeps_all_errors()
{
    const std::vector<int> indices{25, 26};
    const std::vector<std::vector<cv::Point2f>> projected{
        {{0.0F, 0.0F}, {0.0F, 0.0F}},
        {{0.0F, 0.0F}, {0.0F, 0.0F}},
    };
    const std::vector<std::vector<cv::Point2f>> detected{
        {{0.0F, 0.0F}, {3.0F, 0.0F}},
        {{4.0F, 0.0F}, {0.0F, 0.0F}},
    };

    const auto summary = calibration::summarize_projection_errors(
        indices, projected, detected);
    CHECK(summary.images.size() == 2);
    CHECK(summary.images.at(0).max_error_px == 3.0);
    CHECK(summary.images.at(1).max_error_px == 4.0);
    CHECK(summary.global_max_error_px == 4.0);
    CHECK(summary.point_count == 4);
    CHECK(std::abs(summary.rms_error_px - 2.5) < 1e-12);
}

calibration::ProjectionSummary complete_summary(double global_max)
{
    calibration::ProjectionSummary summary;
    summary.global_max_error_px = global_max;
    summary.point_count = calibration::kHeldOutImageCount *
                          calibration::kCornersPerImage;
    for (int image = 0; image < calibration::kHeldOutImageCount; ++image)
    {
        calibration::ImageProjectionErrors errors;
        errors.image_index = calibration::kHeldOutFirstIndex + image;
        errors.point_errors_px.assign(calibration::kCornersPerImage, 0.0);
        errors.max_error_px = global_max;
        summary.images.push_back(errors);
    }
    return summary;
}

void test_gate_uses_compiled_complete_set_and_threshold()
{
    CHECK(calibration::passes_projection_gate(complete_summary(3.0)));
    CHECK(!calibration::passes_projection_gate(complete_summary(3.000001)));
}

void test_invalid_inputs_throw()
{
    const cv::Mat identity = cv::Mat::eye(3, 3, CV_64F);
    const cv::Mat translation = cv::Mat::zeros(3, 1, CV_64F);
    const std::vector<cv::Point3f> point{cv::Point3f(0.0F, 0.0F, 1.0F)};

    check_invalid_argument([&] {
        calibration::color_points_to_thermal(
            point, cv::Mat::eye(2, 2, CV_64F), translation);
    });
    check_invalid_argument([&] {
        calibration::color_points_to_thermal(
            point, identity, cv::Mat::zeros(1, 3, CV_64F));
    });
    check_invalid_argument([&] {
        auto invalid = point;
        invalid[0].x = std::numeric_limits<float>::quiet_NaN();
        calibration::color_points_to_thermal(invalid, identity, translation);
    });
    check_invalid_argument([&] {
        cv::Mat invalid = identity.clone();
        invalid.at<double>(0, 0) = std::numeric_limits<double>::infinity();
        calibration::color_points_to_thermal(point, invalid, translation);
    });

    const std::vector<int> indices{25};
    const std::vector<std::vector<cv::Point2f>> one{{{0.0F, 0.0F}}};
    const std::vector<std::vector<cv::Point2f>> empty_points{{}};
    check_invalid_argument([&] {
        calibration::summarize_projection_errors({25, 26}, one, one);
    });
    check_invalid_argument([&] {
        calibration::summarize_projection_errors(indices, one, empty_points);
    });
    check_invalid_argument([&] {
        calibration::summarize_projection_errors(indices, empty_points, empty_points);
    });
    check_invalid_argument([&] {
        auto non_finite = one;
        non_finite[0][0].x = std::numeric_limits<float>::infinity();
        calibration::summarize_projection_errors(indices, non_finite, one);
    });
}

void test_projection_report_contains_complete_machine_readable_result()
{
    const std::string path = HELDOUT_REPORT_JSON_PATH;
    std::string error;
    CHECK(calibration::write_projection_report_json(
        path, complete_summary(2.5), {}, error));
    CHECK(error.empty());

    cv::FileStorage report(path, cv::FileStorage::READ);
    CHECK(report.isOpened());
    CHECK(static_cast<std::string>(report["schema_version"]) ==
          "thermal-heldout-projection/v1");
    CHECK(static_cast<std::string>(report["status"]) == "pass");
    CHECK(static_cast<int>(report["requested_image_count"]) == 12);
    CHECK(static_cast<int>(report["evaluated_image_count"]) == 12);
    CHECK(static_cast<int>(report["point_count"]) == 144);
    CHECK(std::abs(static_cast<double>(report["threshold_px"]) - 3.0) < 1e-12);
    CHECK(report["images"].isSeq());
    CHECK(report["images"].size() == 12);
    CHECK(report["failures"].isSeq());
    CHECK(report["failures"].size() == 0);
    const cv::FileNode first = report["images"][0];
    CHECK(static_cast<int>(first["index"]) == 25);
    CHECK(first["point_errors_px"].size() == 12);
    CHECK(std::abs(static_cast<double>(first["max_error_px"]) - 2.5) < 1e-12);
}

void test_projection_report_fails_closed()
{
    const std::string root = "/tmp/thermal-heldout-report-test";
    CHECK(mkdir(root.c_str(), 0700) == 0 || errno == EEXIST);
    std::string error;

    const std::vector<calibration::ValidationFailure> validation_failures{
        {25, "detection failed"}};
    CHECK(calibration::write_projection_report_json(
        root + "/failure.json", complete_summary(2.0), validation_failures, error));
    cv::FileStorage failed(root + "/failure.json", cv::FileStorage::READ);
    CHECK(static_cast<std::string>(failed["status"]) == "fail");
    CHECK(failed["failures"].size() == 1);

    calibration::ProjectionSummary incomplete;
    error.clear();
    CHECK(calibration::write_projection_report_json(
        root + "/incomplete.json", incomplete, {}, error));
    cv::FileStorage incomplete_report(
        root + "/incomplete.json", cv::FileStorage::READ);
    CHECK(static_cast<std::string>(incomplete_report["status"]) == "fail");

    error.clear();
    CHECK(!calibration::write_projection_report_json(
        root + "/missing/parent/report.json", complete_summary(2.0), {}, error));
    CHECK(!error.empty());
}

bool parse_options(
    std::vector<std::string> arguments,
    calibration::HeldOutOptions &options,
    std::string &error)
{
    std::vector<char *> argv;
    argv.reserve(arguments.size());
    for (std::string &argument : arguments)
    {
        argv.push_back(&argument[0]);
    }
    return calibration::parse_heldout_options(
        static_cast<int>(argv.size()), argv.data(), options, error);
}

void test_parser_accepts_only_complete_path_contract_or_help()
{
    calibration::HeldOutOptions options;
    std::string error;
    CHECK(parse_options(
        {"heldout_verify", "--color-dir", "color", "--thermal-dir", "thermal",
         "--intrinsic", "intrinsic.xml", "--extrinsic", "extrinsic.xml",
         "--output", "report.json"},
        options,
        error));
    CHECK(!options.help_requested);
    CHECK(options.color_directory == "color");
    CHECK(options.thermal_directory == "thermal");
    CHECK(options.intrinsic_xml == "intrinsic.xml");
    CHECK(options.extrinsic_xml == "extrinsic.xml");
    CHECK(options.output_json == "report.json");

    options = calibration::HeldOutOptions{};
    error.clear();
    CHECK(parse_options({"heldout_verify", "--help"}, options, error));
    CHECK(options.help_requested);
}

void test_parser_rejects_missing_repeated_and_unknown_options()
{
    const std::vector<std::vector<std::string>> invalid{
        {"heldout_verify", "--color-dir"},
        {"heldout_verify", "--color-dir", "color"},
        {"heldout_verify", "--color-dir", "a", "--color-dir", "b",
         "--thermal-dir", "thermal", "--intrinsic", "i", "--extrinsic", "e",
         "--output", "o"},
        {"heldout_verify", "--unknown", "value"},
    };
    for (const auto &arguments : invalid)
    {
        calibration::HeldOutOptions options;
        std::string error;
        CHECK(!parse_options(arguments, options, error));
        CHECK(!error.empty());
    }
}

void test_realsense_distortion_contract_matches_opencv_mapping()
{
    auto intr = [](rs2_distortion model, std::array<float, 5> coeffs) {
        rs2_intrinsics i{};
        i.model = model;
        for (int k = 0; k < 5; ++k) i.coeffs[k] = coeffs[k];
        return i;
    };
    // NONE and BROWN_CONRADY (with real coeffs) are accepted.
    CHECK(calibration::supported_realsense_color_distortion(
        intr(RS2_DISTORTION_NONE, {0, 0, 0, 0, 0})));
    CHECK(calibration::supported_realsense_color_distortion(
        intr(RS2_DISTORTION_BROWN_CONRADY, {0.1F, -0.2F, 0, 0, 0.05F})));
    // INVERSE_BROWN_CONRADY accepted ONLY when every coefficient is zero.
    CHECK(calibration::supported_realsense_color_distortion(
        intr(RS2_DISTORTION_INVERSE_BROWN_CONRADY, {0, 0, 0, 0, 0})));
    CHECK(!calibration::supported_realsense_color_distortion(
        intr(RS2_DISTORTION_INVERSE_BROWN_CONRADY, {0.01F, 0, 0, 0, 0})));
    CHECK(!calibration::supported_realsense_color_distortion(
        intr(RS2_DISTORTION_INVERSE_BROWN_CONRADY, {0, 0, 0, 0.02F, 0})));
    // Other models are rejected outright.
    CHECK(!calibration::supported_realsense_color_distortion(
        intr(RS2_DISTORTION_KANNALA_BRANDT4, {0, 0, 0, 0, 0})));
}

}  // namespace

int main()
{
    test_transform_inverts_non_identity_thermal_to_color_pose();
    test_projection_summary_keeps_all_errors();
    test_gate_uses_compiled_complete_set_and_threshold();
    test_invalid_inputs_throw();
    test_projection_report_contains_complete_machine_readable_result();
    test_projection_report_fails_closed();
    test_parser_accepts_only_complete_path_contract_or_help();
    test_parser_rejects_missing_repeated_and_unknown_options();
    test_realsense_distortion_contract_matches_opencv_mapping();

    if (failures != 0)
    {
        std::cerr << failures << " heldout verifier checks failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "All heldout verifier checks passed\n";
    return EXIT_SUCCESS;
}
