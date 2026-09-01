/*
 * SPI testing utility (using spidev driver)
 *
 * Copyright (c) 2007  MontaVista Software, Inc.
 * Copyright (c) 2007  Anton Vorontsov <avorontsov@ru.mvista.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License.
 *
 * Cross-compile with cross-gcc -I/path/to/cross-kernel/include
 */

#ifndef SPI_H
#define SPI_H

/// \file spi.h
/// \brief SPI communication between the RaspberryPi and Lepton 3.1R thermal camera.

#include <string>
#include <stdint.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/types.h>
#include <linux/spi/spidev.h>

/// \brief Descriptor for SPI channel 0.
extern int spi_cs0_fd;
/// \brief Descriptor for SPI channel 1.
extern int spi_cs1_fd;
/// \brief Descriptor for SPI mode.
extern unsigned char spi_mode;
/// \brief Descriptor for number of bits per word used in SPI communication.
extern unsigned char spi_bitsPerWord;
/// \brief Descriptor for SPI communication speed.
extern unsigned int spi_speed;

/// \brief Opens the SPI port communcation.
/// \param spi_device SPI device to be opened (0 or 1).
/// \param spi_speed Speed of SPI communication.
/// \return 0 if successful, -1 if failure.
int SpiOpenPort(int spi_device, unsigned int spi_speed);

/// \brief Closes the SPI port communication.
/// \param spi_device SPI device to be closed (0 or 1).
/// \return 0 if successful, -1 if failure
int SpiClosePort(int spi_device);

#endif