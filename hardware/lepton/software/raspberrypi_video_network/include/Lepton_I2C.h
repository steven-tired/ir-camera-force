#ifndef LEPTON_I2C
#define LEPTON_I2C

/// \file lepton_i2c.h
/// \brief The I2C communication functions for the Lepton 3.1R thermal camera.

/// \brief Connects to the Lepton I2C port.
/// \return 0 if successful.
int lepton_connect();

/// Configure verified Raw14 cooked TLinear (0.01 K), manual FFC, and footer telemetry.
/// Returns true only when every setting can be read back successfully.
bool lepton_configure_tlinear_telemetry();

/// Verify the streaming configuration without changing video or telemetry state.
/// This is safe to call from a separate CCI process while the SPI streamer runs.
bool lepton_verify_tlinear_telemetry();

/// \brief Performs the Flat Field Correction (FFC) on the Lepton camera.
/// \param timeout_ms Maximum time to wait for the camera to report ready.
/// \return true when FFC completes before the timeout.
bool lepton_perform_ffc(unsigned int timeout_ms = 5000);

/// \brief Reboots the Lepton camera, resetting internal states.
bool lepton_reboot();

#endif
