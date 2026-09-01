#include <cstdio>
#include <cstring>
#include <unistd.h>

extern "C" {
#include "LEPTON_SDK.h"
#include "LEPTON_SYS.h"
#include "LEPTON_OEM.h"
}

static void print_result(const char *name, LEP_RESULT r) {
    std::printf("%-36s result=%d\n", name, static_cast<int>(r));
}

static bool ok(LEP_RESULT r) {
    return r == LEP_OK;
}

int main(int argc, char **argv) {
    bool fix_video = false;
    bool reboot = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--fix-video") == 0) {
            fix_video = true;
        } else if (std::strcmp(argv[i], "--reboot") == 0) {
            reboot = true;
        } else {
            std::fprintf(stderr, "Usage: %s [--fix-video] [--reboot]\n", argv[0]);
            return 2;
        }
    }

    LEP_CAMERA_PORT_DESC_T port;
    LEP_RESULT r = LEP_OpenPort(1, LEP_CCI_TWI, 400, &port);
    print_result("LEP_OpenPort", r);
    if (!ok(r)) {
        return 1;
    }

    r = LEP_RunSysPing(&port);
    print_result("LEP_RunSysPing", r);

    LEP_STATUS_T sys_status{};
    r = LEP_GetSysStatus(&port, &sys_status);
    print_result("LEP_GetSysStatus", r);
    if (ok(r)) {
        std::printf("  camStatus=%d commandCount=%u\n",
                    static_cast<int>(sys_status.camStatus),
                    static_cast<unsigned>(sys_status.commandCount));
    }

    LEP_SYS_AUX_TEMPERATURE_KELVIN_T aux_k = 0;
    r = LEP_GetSysAuxTemperatureKelvin(&port, &aux_k);
    print_result("LEP_GetSysAuxTemperatureKelvin", r);
    if (ok(r)) {
        std::printf("  aux_kelvin_x100=%u aux_c=%.2f\n",
                    static_cast<unsigned>(aux_k), aux_k / 100.0 - 273.15);
    }

    LEP_SYS_FPA_TEMPERATURE_KELVIN_T fpa_k = 0;
    r = LEP_GetSysFpaTemperatureKelvin(&port, &fpa_k);
    print_result("LEP_GetSysFpaTemperatureKelvin", r);
    if (ok(r)) {
        std::printf("  fpa_kelvin_x100=%u fpa_c=%.2f\n",
                    static_cast<unsigned>(fpa_k), fpa_k / 100.0 - 273.15);
    }

    LEP_SYS_TELEMETRY_ENABLE_STATE_E telemetry{};
    r = LEP_GetSysTelemetryEnableState(&port, &telemetry);
    print_result("LEP_GetSysTelemetryEnableState", r);
    if (ok(r)) {
        std::printf("  telemetry=%d (0=disabled, 1=enabled)\n", static_cast<int>(telemetry));
    }

    LEP_OEM_POWER_STATE_E power{};
    r = LEP_GetOemPowerMode(&port, &power);
    print_result("LEP_GetOemPowerMode", r);
    if (ok(r)) {
        std::printf("  power=%d (0=normal)\n", static_cast<int>(power));
    }

    LEP_OEM_VIDEO_OUTPUT_ENABLE_E video_enable{};
    r = LEP_GetOemVideoOutputEnable(&port, &video_enable);
    print_result("LEP_GetOemVideoOutputEnable", r);
    if (ok(r)) {
        std::printf("  video_enable=%d (1=enabled)\n", static_cast<int>(video_enable));
    }

    LEP_OEM_VIDEO_OUTPUT_FORMAT_E video_format{};
    r = LEP_GetOemVideoOutputFormat(&port, &video_format);
    print_result("LEP_GetOemVideoOutputFormat", r);
    if (ok(r)) {
        std::printf("  video_format=%d (7=RAW14)\n", static_cast<int>(video_format));
    }

    LEP_OEM_VIDEO_OUTPUT_SOURCE_E video_source{};
    r = LEP_GetOemVideoOutputSource(&port, &video_source);
    print_result("LEP_GetOemVideoOutputSource", r);
    if (ok(r)) {
        std::printf("  video_source=%d (0=RAW)\n", static_cast<int>(video_source));
    }

    LEP_OEM_VIDEO_OUTPUT_CHANNEL_E video_channel{};
    r = LEP_GetOemVideoOutputChannel(&port, &video_channel);
    print_result("LEP_GetOemVideoOutputChannel", r);
    if (ok(r)) {
        std::printf("  video_channel=%d (1=VoSPI)\n", static_cast<int>(video_channel));
    }

    if (fix_video) {
        std::puts("\nApplying runtime video settings for raspberrypi_video_network...");
        print_result("LEP_SetOemPowerMode(NORMAL)",
                     LEP_SetOemPowerMode(&port, LEP_OEM_POWER_MODE_NORMAL));
        print_result("LEP_SetSysTelemetryEnableState(DISABLED)",
                     LEP_SetSysTelemetryEnableState(&port, LEP_TELEMETRY_DISABLED));
        print_result("LEP_SetOemVideoOutputEnable(ENABLE)",
                     LEP_SetOemVideoOutputEnable(&port, LEP_VIDEO_OUTPUT_ENABLE));
        print_result("LEP_SetOemVideoOutputFormat(RAW14)",
                     LEP_SetOemVideoOutputFormat(&port, LEP_VIDEO_OUTPUT_FORMAT_RAW14));
        print_result("LEP_SetOemVideoOutputSource(RAW)",
                     LEP_SetOemVideoOutputSource(&port, LEP_VIDEO_OUTPUT_SOURCE_RAW));
        print_result("LEP_SetOemVideoOutputChannel(VOSPI)",
                     LEP_SetOemVideoOutputChannel(&port, LEP_VIDEO_OUTPUT_CHANNEL_VOSPI));
    }

    if (reboot) {
        std::puts("\nRebooting Lepton over I2C...");
        print_result("LEP_RunOemReboot", LEP_RunOemReboot(&port));
        sleep(3);
    }

    LEP_ClosePort(&port);
    return 0;
}
