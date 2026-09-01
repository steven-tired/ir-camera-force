#include "Lepton_I2C.h"

#include "LEPTON_SDK.h"
#include "LEPTON_AGC.h"
#include "LEPTON_SYS.h"
#include "LEPTON_OEM.h"
#include "LEPTON_Types.h"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <thread>

/// \brief private variable, descriptor for connection to Lepton I2C port.
bool _connected;

/// \brief private variable, descriptor for Lepton I2C port struct.
LEP_CAMERA_PORT_DESC_T _port;

namespace
{
// Teledyne FLIR Lepton SDK main@6f92303, LEPTON_RAD.h. The bundled SDK is
// older and lacks the RAD module, so use the official typed command layout.
constexpr LEP_COMMAND_ID kRadEnableState = 0x4E10;
constexpr LEP_COMMAND_ID kRadTLinearEnableState = 0x4EC0;
constexpr LEP_COMMAND_ID kRadTLinearResolution = 0x4EC4;
constexpr LEP_COMMAND_ID kRadTLinearAutoResolution = 0x4EC8;

bool check(LEP_RESULT result, const char *operation)
{
    if (result == LEP_OK)
    {
        return true;
    }
    std::cerr << operation << " failed: LEP_RESULT=" << result << std::endl;
    return false;
}

bool set_rad_enum(LEP_COMMAND_ID command, std::uint32_t value, const char *operation)
{
    return check(
        LEP_SetAttribute(&_port, command, reinterpret_cast<LEP_ATTRIBUTE_T_PTR>(&value), 2),
        operation);
}

bool get_rad_enum(LEP_COMMAND_ID command, std::uint32_t *value, const char *operation)
{
    return check(
        LEP_GetAttribute(&_port, command, reinterpret_cast<LEP_ATTRIBUTE_T_PTR>(value), 2),
        operation);
}

bool verify_tlinear_telemetry()
{
    LEP_AGC_ENABLE_E agc = LEP_AGC_ENABLE;
    LEP_OEM_VIDEO_OUTPUT_SOURCE_E source = LEP_VIDEO_OUTPUT_SOURCE_RAW;
    LEP_OEM_VIDEO_OUTPUT_FORMAT_E format = LEP_VIDEO_OUTPUT_FORMAT_RAW8;
    LEP_OEM_VIDEO_OUTPUT_ENABLE_E video = LEP_VIDEO_OUTPUT_DISABLE;
    LEP_SYS_TELEMETRY_LOCATION_E location = LEP_TELEMETRY_LOCATION_HEADER;
    LEP_SYS_TELEMETRY_ENABLE_STATE_E telemetry = LEP_TELEMETRY_DISABLED;
    LEP_SYS_FFC_SHUTTER_MODE_OBJ_T shutter = {};
    std::uint32_t rad = 0;
    std::uint32_t tlinear = 0;
    std::uint32_t resolution = 0;
    std::uint32_t auto_resolution = 1;

    bool ok = true;
    ok &= check(LEP_GetAgcEnableState(&_port, &agc), "verify AGC state");
    ok &= check(LEP_GetOemVideoOutputSource(&_port, &source), "verify video source");
    ok &= check(LEP_GetOemVideoOutputFormat(&_port, &format), "verify video format");
    ok &= check(LEP_GetOemVideoOutputEnable(&_port, &video), "verify video output");
    ok &= get_rad_enum(kRadEnableState, &rad, "verify radiometry");
    ok &= get_rad_enum(kRadTLinearEnableState, &tlinear, "verify TLinear");
    ok &= get_rad_enum(kRadTLinearResolution, &resolution, "verify TLinear resolution");
    ok &= get_rad_enum(
        kRadTLinearAutoResolution, &auto_resolution, "verify TLinear auto resolution");
    ok &= check(LEP_GetSysFfcShutterModeObj(&_port, &shutter), "verify FFC mode");
    ok &= check(LEP_GetSysTelemetryLocation(&_port, &location), "verify telemetry location");
    ok &= check(LEP_GetSysTelemetryEnableState(&_port, &telemetry), "verify telemetry state");
    ok &= agc == LEP_AGC_DISABLE;
    ok &= source == LEP_VIDEO_OUTPUT_SOURCE_COOKED;
    ok &= format == LEP_VIDEO_OUTPUT_FORMAT_RAW14;
    ok &= video == LEP_VIDEO_OUTPUT_ENABLE;
    ok &= rad == 1 && tlinear == 1 && resolution == 1 && auto_resolution == 0;
    ok &= shutter.shutterMode == LEP_SYS_FFC_SHUTTER_MODE_MANUAL;
    ok &= location == LEP_TELEMETRY_LOCATION_FOOTER;
    ok &= telemetry == LEP_TELEMETRY_ENABLED;
    if (!ok)
    {
        std::cerr << "Lepton configuration read-back mismatch" << std::endl;
        return false;
    }
    std::cout << "Lepton verified: AGC=disabled, Raw14 cooked TLinear=0.01K, "
                 "manual FFC, telemetry footer"
              << std::endl;
    return true;
}
} // namespace

int lepton_connect() {
	if (_connected) {
		return LEP_OK;
	}
	LEP_RESULT result = LEP_OpenPort(1, LEP_CCI_TWI, 400, &_port);
	_connected = result == LEP_OK;
	return result;
}

bool lepton_configure_tlinear_telemetry()
{
    if (!check(static_cast<LEP_RESULT>(lepton_connect()), "LEP_OpenPort"))
    {
        return false;
    }

    LEP_SYS_FFC_SHUTTER_MODE_OBJ_T shutter = {};
    if (!check(LEP_GetSysFfcShutterModeObj(&_port, &shutter), "get FFC shutter mode"))
    {
        return false;
    }
    shutter.shutterMode = LEP_SYS_FFC_SHUTTER_MODE_MANUAL;

    bool ok = true;
    ok &= check(LEP_SetAgcEnableState(&_port, LEP_AGC_DISABLE), "disable AGC");
    ok &= check(
        LEP_SetOemVideoOutputSource(&_port, LEP_VIDEO_OUTPUT_SOURCE_COOKED),
        "set cooked video source");
    ok &= check(
        LEP_SetOemVideoOutputFormat(&_port, LEP_VIDEO_OUTPUT_FORMAT_RAW14),
        "set Raw14 video format");
    ok &= set_rad_enum(kRadEnableState, 1, "enable radiometry");
    ok &= set_rad_enum(kRadTLinearAutoResolution, 0, "disable TLinear auto resolution");
    ok &= set_rad_enum(kRadTLinearResolution, 1, "set TLinear resolution to 0.01 K");
    ok &= set_rad_enum(kRadTLinearEnableState, 1, "enable TLinear");
    ok &= check(LEP_SetSysFfcShutterModeObj(&_port, shutter), "set manual FFC mode");
    ok &= check(
        LEP_SetSysTelemetryLocation(&_port, LEP_TELEMETRY_LOCATION_FOOTER),
        "set telemetry footer");
    ok &= check(
        LEP_SetSysTelemetryEnableState(&_port, LEP_TELEMETRY_ENABLED),
        "enable telemetry");
    ok &= check(
        LEP_SetOemVideoOutputEnable(&_port, LEP_VIDEO_OUTPUT_ENABLE),
        "enable video output");
    if (!ok)
    {
        return false;
    }
    return verify_tlinear_telemetry();
}

bool lepton_verify_tlinear_telemetry()
{
    if (!check(static_cast<LEP_RESULT>(lepton_connect()), "LEP_OpenPort"))
    {
        return false;
    }
    return verify_tlinear_telemetry();
}

bool lepton_perform_ffc(unsigned int timeout_ms) {
	if (!check(static_cast<LEP_RESULT>(lepton_connect()), "LEP_OpenPort")) {
		return false;
	}
	if (!check(
			LEP_RunCommand(&_port, static_cast<LEP_COMMAND_ID>(FLR_CID_SYS_RUN_FFC)),
			"run FFC")) {
		return false;
	}
	const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
	while (std::chrono::steady_clock::now() < deadline) {
		LEP_SYS_STATUS_E status = LEP_SYS_STATUS_BUSY;
		if (!check(LEP_GetSysFFCStatus(&_port, &status), "get FFC status")) {
			return false;
		}
		if (status == LEP_SYS_STATUS_READY) {
			std::cout << "Manual FFC complete" << std::endl;
			return true;
		}
		if (status != LEP_SYS_STATUS_BUSY) {
			std::cerr << "Manual FFC failed: status=" << status << std::endl;
			return false;
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}
	std::cerr << "Manual FFC timed out after " << timeout_ms << " ms" << std::endl;
	return false;
}

bool lepton_reboot() {
	if (!check(static_cast<LEP_RESULT>(lepton_connect()), "LEP_OpenPort")) {
		return false;
	}
	const bool ok = check(LEP_RunOemReboot(&_port), "reboot Lepton");
	LEP_ClosePort(&_port);
	_connected = false;
	return ok;
}
