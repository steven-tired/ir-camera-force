#include "HeldOutVerifier.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace calibration {
namespace {

bool finite_point(const cv::Point3f &point)
{
    return std::isfinite(point.x) && std::isfinite(point.y) &&
           std::isfinite(point.z);
}

bool finite_point(const cv::Point2f &point)
{
    return std::isfinite(point.x) && std::isfinite(point.y);
}

cv::Mat as_finite_double_matrix(
    const cv::Mat &input, int rows, int columns, const char *name)
{
    if (input.rows != rows || input.cols != columns || input.channels() != 1)
    {
        throw std::invalid_argument(std::string(name) + " has an invalid shape");
    }
    cv::Mat converted;
    input.convertTo(converted, CV_64F);
    if (!cv::checkRange(converted, true, nullptr))
    {
        throw std::invalid_argument(std::string(name) + " contains non-finite values");
    }
    return converted;
}

}  // namespace

std::vector<cv::Point3f> color_points_to_thermal(
    const std::vector<cv::Point3f> &color_points,
    const cv::Mat &rotation_thermal_to_color,
    const cv::Mat &translation_thermal_to_color)
{
    if (color_points.empty())
    {
        throw std::invalid_argument("Color point set must not be empty");
    }
    const cv::Mat rotation = as_finite_double_matrix(
        rotation_thermal_to_color, 3, 3, "Rotation");
    const cv::Mat translation = as_finite_double_matrix(
        translation_thermal_to_color, 3, 1, "Translation");

    std::vector<cv::Point3f> thermal_points;
    thermal_points.reserve(color_points.size());
    for (const cv::Point3f &point : color_points)
    {
        if (!finite_point(point))
        {
            throw std::invalid_argument("Color point contains non-finite values");
        }
        const cv::Mat x_color =
            (cv::Mat_<double>(3, 1) << point.x, point.y, point.z);
        const cv::Mat x_thermal = rotation.t() * (x_color - translation);
        thermal_points.emplace_back(
            static_cast<float>(x_thermal.at<double>(0, 0)),
            static_cast<float>(x_thermal.at<double>(1, 0)),
            static_cast<float>(x_thermal.at<double>(2, 0)));
    }
    return thermal_points;
}

ProjectionSummary summarize_projection_errors(
    const std::vector<int> &image_indices,
    const std::vector<std::vector<cv::Point2f>> &projected,
    const std::vector<std::vector<cv::Point2f>> &detected)
{
    if (image_indices.empty() || image_indices.size() != projected.size() ||
        projected.size() != detected.size())
    {
        throw std::invalid_argument("Image vectors must be non-empty and equally sized");
    }

    ProjectionSummary summary;
    double sum_squared_error = 0.0;
    for (std::size_t image = 0; image < image_indices.size(); ++image)
    {
        if (projected[image].empty() ||
            projected[image].size() != detected[image].size())
        {
            throw std::invalid_argument(
                "Projected and detected point sets must be non-empty and equally sized");
        }

        ImageProjectionErrors image_errors;
        image_errors.image_index = image_indices[image];
        image_errors.point_errors_px.reserve(projected[image].size());
        for (std::size_t point = 0; point < projected[image].size(); ++point)
        {
            if (!finite_point(projected[image][point]) ||
                !finite_point(detected[image][point]))
            {
                throw std::invalid_argument("Projection point contains non-finite values");
            }
            const double error = cv::norm(projected[image][point] - detected[image][point]);
            image_errors.point_errors_px.push_back(error);
            image_errors.max_error_px = std::max(image_errors.max_error_px, error);
            summary.global_max_error_px = std::max(summary.global_max_error_px, error);
            sum_squared_error += error * error;
            ++summary.point_count;
        }
        summary.images.push_back(image_errors);
    }
    summary.rms_error_px = std::sqrt(sum_squared_error / summary.point_count);
    return summary;
}

bool passes_projection_gate(const ProjectionSummary &summary)
{
    if (summary.images.size() != kHeldOutImageCount ||
        summary.point_count != static_cast<std::size_t>(
            kHeldOutImageCount * kCornersPerImage) ||
        !std::isfinite(summary.global_max_error_px) ||
        !std::isfinite(summary.rms_error_px) ||
        summary.global_max_error_px > kHeldOutThresholdPx)
    {
        return false;
    }
    for (const ImageProjectionErrors &image : summary.images)
    {
        if (image.point_errors_px.size() != kCornersPerImage ||
            !std::isfinite(image.max_error_px))
        {
            return false;
        }
        for (double error : image.point_errors_px)
        {
            if (!std::isfinite(error))
            {
                return false;
            }
        }
    }
    return true;
}

bool write_projection_report_json(
    const std::string &path,
    const ProjectionSummary &summary,
    const std::vector<ValidationFailure> &failures,
    std::string &error)
{
    error.clear();
    try
    {
        cv::FileStorage storage(
            path, cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);
        if (!storage.isOpened())
        {
            error = "Failed to open report for writing: " + path;
            return false;
        }

        const bool passed = failures.empty() && passes_projection_gate(summary);
        storage << "schema_version" << "thermal-heldout-projection/v1";
        storage << "status" << (passed ? "pass" : "fail");
        storage << "threshold_px" << kHeldOutThresholdPx;
        storage << "requested_image_count" << kHeldOutImageCount;
        storage << "evaluated_image_count" << static_cast<int>(summary.images.size());
        storage << "point_count" << static_cast<int>(summary.point_count);
        storage << "global_max_error_px" << summary.global_max_error_px;
        storage << "rms_error_px" << summary.rms_error_px;

        storage << "images" << "[";
        for (const ImageProjectionErrors &image : summary.images)
        {
            storage << "{";
            storage << "index" << image.image_index;
            storage << "point_errors_px" << "[";
            for (double point_error : image.point_errors_px)
            {
                storage << point_error;
            }
            storage << "]";
            storage << "max_error_px" << image.max_error_px;
            storage << "}";
        }
        storage << "]";

        storage << "failures" << "[";
        for (const ValidationFailure &failure : failures)
        {
            storage << "{";
            storage << "image_index" << failure.image_index;
            storage << "reason" << failure.reason;
            storage << "}";
        }
        storage << "]";
        storage.release();
        return true;
    }
    catch (const cv::Exception &exception)
    {
        error = exception.what();
        return false;
    }
}

bool parse_heldout_options(
    int argc, char **argv, HeldOutOptions &options, std::string &error)
{
    options = HeldOutOptions{};
    error.clear();
    if (argc == 2 && std::string(argv[1]) == "--help")
    {
        options.help_requested = true;
        return true;
    }

    struct OptionBinding
    {
        const char *name;
        std::string HeldOutOptions::*field;
    };
    const OptionBinding bindings[] = {
        {"--color-dir", &HeldOutOptions::color_directory},
        {"--thermal-dir", &HeldOutOptions::thermal_directory},
        {"--intrinsic", &HeldOutOptions::intrinsic_xml},
        {"--extrinsic", &HeldOutOptions::extrinsic_xml},
        {"--output", &HeldOutOptions::output_json},
    };

    for (int index = 1; index < argc; ++index)
    {
        const std::string argument(argv[index]);
        const OptionBinding *binding = nullptr;
        for (const OptionBinding &candidate : bindings)
        {
            if (argument == candidate.name)
            {
                binding = &candidate;
                break;
            }
        }
        if (binding == nullptr)
        {
            error = "Unknown option: " + argument;
            return false;
        }
        std::string &destination = options.*(binding->field);
        if (!destination.empty())
        {
            error = "Repeated option: " + argument;
            return false;
        }
        if (index + 1 >= argc || std::string(argv[index + 1]).compare(0, 2, "--") == 0)
        {
            error = "Missing value for option: " + argument;
            return false;
        }
        destination = argv[++index];
    }

    for (const OptionBinding &binding : bindings)
    {
        if ((options.*(binding.field)).empty())
        {
            error = std::string("Missing required option: ") + binding.name;
            return false;
        }
    }
    return true;
}

bool supported_realsense_color_distortion(const rs2_intrinsics &intrinsics)
{
    if (intrinsics.model == RS2_DISTORTION_NONE ||
        intrinsics.model == RS2_DISTORTION_BROWN_CONRADY)
    {
        return true;
    }
    if (intrinsics.model == RS2_DISTORTION_INVERSE_BROWN_CONRADY)
    {
        for (int i = 0; i < 5; ++i)
        {
            if (intrinsics.coeffs[i] != 0.0F)
            {
                return false;
            }
        }
        return true;
    }
    return false;
}

}  // namespace calibration
