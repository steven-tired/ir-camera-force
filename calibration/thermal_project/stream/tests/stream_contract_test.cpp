#include "RealSenseCaptureContract.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

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

thermal_stream::RealSenseCaptureProfile exact_profile()
{
    return {
        thermal_stream::kExpectedRealSenseSerial,
        1280,
        720,
        RS2_FORMAT_RGB8,
        15,
        1280,
        720,
        RS2_FORMAT_Z16,
        6,
    };
}

void test_exact_capture_contract_and_each_mismatch()
{
    CHECK(std::string(thermal_stream::kExpectedRealSenseSerial) == "233522078685");
    CHECK(thermal_stream::matches_calibration_capture_contract(exact_profile()));

    auto profile = exact_profile();
    profile.serial = "wrong";
    CHECK(!thermal_stream::matches_calibration_capture_contract(profile));
    profile = exact_profile();
    profile.color_width = 640;
    CHECK(!thermal_stream::matches_calibration_capture_contract(profile));
    profile = exact_profile();
    profile.color_format = RS2_FORMAT_BGR8;
    CHECK(!thermal_stream::matches_calibration_capture_contract(profile));
    profile = exact_profile();
    profile.color_fps = 30;
    CHECK(!thermal_stream::matches_calibration_capture_contract(profile));
    profile = exact_profile();
    profile.depth_height = 480;
    CHECK(!thermal_stream::matches_calibration_capture_contract(profile));
    profile = exact_profile();
    profile.depth_format = RS2_FORMAT_DISPARITY16;
    CHECK(!thermal_stream::matches_calibration_capture_contract(profile));
    profile = exact_profile();
    profile.depth_fps = 15;
    CHECK(!thermal_stream::matches_calibration_capture_contract(profile));
}

void test_capture_source_binds_device_and_logs_exact_contract()
{
    std::ifstream source(DEPTHIMAGE_SOURCE_PATH);
    std::ostringstream contents;
    contents << source.rdbuf();
    const std::string text = contents.str();
    CHECK(text.find("enable_device(thermal_stream::kExpectedRealSenseSerial)") !=
          std::string::npos);
    CHECK(text.find(
        "RealSense capture contract: serial=233522078685 "
        "color=1280x720 RGB8@15 depth=1280x720 Z16@6") != std::string::npos);
}

}  // namespace

int main()
{
    test_exact_capture_contract_and_each_mismatch();
    test_capture_source_binds_device_and_logs_exact_contract();
    if (failures != 0)
    {
        std::cerr << failures << " stream contract checks failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "stream_contract_test: PASS\n";
    return EXIT_SUCCESS;
}
