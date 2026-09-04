#include "RealSenseCaptureContract.h"

namespace thermal_stream {

bool matches_calibration_capture_contract(const RealSenseCaptureProfile &profile)
{
    return profile.serial == kExpectedRealSenseSerial &&
           profile.color_width == 1280 &&
           profile.color_height == 720 &&
           profile.color_format == RS2_FORMAT_RGB8 &&
           profile.color_fps == 15 &&
           profile.depth_width == 1280 &&
           profile.depth_height == 720 &&
           profile.depth_format == RS2_FORMAT_Z16 &&
           profile.depth_fps == 6;
}

}  // namespace thermal_stream
