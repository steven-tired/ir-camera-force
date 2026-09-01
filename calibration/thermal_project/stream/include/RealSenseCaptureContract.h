#pragma once

#include <librealsense2/h/rs_sensor.h>

#include <string>

namespace thermal_stream {

constexpr char kExpectedRealSenseSerial[] = "233522078685";

struct RealSenseCaptureProfile
{
    std::string serial;
    int color_width;
    int color_height;
    rs2_format color_format;
    int color_fps;
    int depth_width;
    int depth_height;
    rs2_format depth_format;
    int depth_fps;
};

bool matches_calibration_capture_contract(const RealSenseCaptureProfile &profile);

}  // namespace thermal_stream
