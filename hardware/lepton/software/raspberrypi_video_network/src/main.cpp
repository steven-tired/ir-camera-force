#include <iostream>
#include <ctime>
#include <stdint.h>
#include <chrono>
#include <unistd.h>
#include <cstdint>
#include <cstring>
#include <arpa/inet.h>

#include "SPI.h"
#include "Lepton_I2C.h"
#include "LeptonVoSPI.h"

namespace
{
constexpr int kBadPacketLimit = 750;
constexpr int kSoftResyncsBeforeReboot = 5;
constexpr useconds_t kVoSPIResyncDelayUs = 200000;
}

/// \file main.cpp
/// \brief Main loop that sends raw data through ethernet using UDP protocol.

/// \brief Function to describe how to use the command line arguments
/// \param cmd Argument of the command line, here it is the program
void printUsage(char *cmd)
{
	char *cmdname = basename(cmd);
	printf("Usage: %s [OPTION]...\n"
		   " -h			display this help and exit\n"
		   " -net x		set the ip address (default: 10.42.0.1)\n"
		   " -port x	set the ip port (default: 8080)\n"
		   " -ffc-only verify the running configuration, run manual FFC, and exit\n"
		   "", cmdname);
	return;
}

/// \brief Main function that captures raw data from the Lepton and transmits it using UDP.
/// \param argc Number of command-line arguments.
/// \param argv Array of command-line arguments.
/// \param net IP address to send the UDP messages.
/// \param port Port of the IP address.
/// \return 0 if successful, -1 if failure.
int main(int argc, char **argv)
{
	unsigned int spiSpeed = 20 * 1000 * 1000; // SPI bus speed 20MHz

	uint8_t result[lepton_vospi::kTelemetrySegmentBytes];
	uint8_t shelf[lepton_vospi::kSegmentCount][lepton_vospi::kTelemetrySegmentBytes];

	uint16_t n_wrong_segment = 0;

	int sockfd;
	struct sockaddr_in servaddr;
	const char *netIP = "10.42.0.1";
	uint16_t port = 8080;
	bool ffcOnly = false;

	for(int i=1; i < argc; i++)
	{
		if (strcmp(argv[i], "-h") == 0)
		{
			printUsage(argv[0]);
			exit(0);
		}
		else if (strcmp(argv[i], "-net") == 0)
		{
			if (i + 1 != argc)
			{
				netIP = argv[++i];
			} else {
				std::cerr << "Error: Enter a valid IP." << std::endl;
				exit(1);
			}
		}
		else if (strcmp(argv[i], "-port") == 0)
		{
			if (i + 1 != argc)
			{
				long int temp = std::strtol(argv[++i], nullptr, 10);
				if (temp < 0 || temp > 65535){
					std::cerr << "Error: Enter a valid Port." << std::endl;
				}
				port = static_cast<uint16_t>(temp);
			} else {
				std::cerr << "Error: Enter a valid Port." << std::endl;
				exit(1);
			}
		}
		else if (strcmp(argv[i], "-ffc-only") == 0)
		{
			ffcOnly = true;
		}
	}
	if (ffcOnly)
	{
		return lepton_verify_tlinear_telemetry() && lepton_perform_ffc() ? 0 : -1;
	}
	if (!lepton_configure_tlinear_telemetry() || !lepton_perform_ffc())
	{
		return -1;
	}
	std::cout << "Network address is " << netIP << std::endl;
	std::cout << "Network port is " << port << std::endl;
	std::cout << "UDP wire format is 4 x " << lepton_vospi::kTelemetrySegmentBytes
			  << " byte telemetry-footer segments" << std::endl;
	// FLIR VoSPI requires chip select to remain deasserted for at least 185 ms
	// before attempting to acquire a new packet-zero boundary.
	usleep(kVoSPIResyncDelayUs);
	SpiOpenPort(0, spiSpeed);

	// Setting up UDP socket
	if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0)
	{
		std::cerr << "Socket creation failed" << std::endl;
		return -1;
	}

	memset(&servaddr, 0, sizeof(servaddr));

	// Set up server address
	servaddr.sin_family = AF_INET;
	servaddr.sin_port = htons(port);
	servaddr.sin_addr.s_addr = inet_addr(netIP);

	int expectedSegment = 1;
	int softResyncs = 0;
	while(true)
	{
		//read data packets from lepton over SPI
		int resets = 0;
		int segmentNumber = -1;
		for(std::size_t j = 0; j < lepton_vospi::kTelemetryPacketsPerSegment; ++j)
		{
			//if it's a drop packet, reset j to 0, set to -1 so he'll be at 0 again loop
			const ssize_t received = read(
				spi_cs0_fd,
				result + lepton_vospi::kPacketSize * j,
				lepton_vospi::kPacketSize);
			int packetNumber = result[j * lepton_vospi::kPacketSize + 1];
			if(received != static_cast<ssize_t>(lepton_vospi::kPacketSize)
				|| packetNumber != static_cast<int>(j))
			{
				j = static_cast<std::size_t>(-1);
				resets += 1;
				usleep(1000);
				//Note: we've selected 750 resets as an arbitrary limit, since there should never be 750 "null" packets between two valid transmissions at the current poll rate
				//By polling faster, developers may easily exceed this count, and the down period between frames may then be flagged as a loss of sync
				if(resets == kBadPacketLimit)
				{
					SpiClosePort(0);
					n_wrong_segment = 0;
					expectedSegment = 1;
					resets = 0;
					softResyncs++;
					std::cerr << "VoSPI sync lost; soft resync " << softResyncs << "/"
							  << kSoftResyncsBeforeReboot << std::endl;
					usleep(kVoSPIResyncDelayUs);
					if (softResyncs < kSoftResyncsBeforeReboot)
					{
						SpiOpenPort(0, spiSpeed);
						continue;
					}

					std::cerr << "VoSPI soft resyncs exhausted; rebooting Lepton" << std::endl;
					if (!lepton_reboot())
					{
						std::cerr << "Lepton reboot failed; returning to soft resync" << std::endl;
						softResyncs = 0;
						usleep(1000000);
						SpiOpenPort(0, spiSpeed);
						continue;
					}
					usleep(1500000);
					if (!lepton_configure_tlinear_telemetry() || !lepton_perform_ffc())
					{
						return -1;
					}
					usleep(kVoSPIResyncDelayUs);
					SpiOpenPort(0, spiSpeed);
					softResyncs = 0;
				}
				continue;
			}
			if (packetNumber == 20)
			{
				segmentNumber = (result[j * lepton_vospi::kPacketSize] >> 4) & 0x0f;
				if ((segmentNumber < 1) || (4 < segmentNumber))
				{
					break;
				}
			}
		}

		if ((segmentNumber < 1) || (4 < segmentNumber))
		{
			n_wrong_segment++;
			continue;
		}
		if (n_wrong_segment != 0)
		{
			n_wrong_segment = 0;
		}
		if (!lepton_vospi::validate_segment(result, sizeof(result)))
		{
			expectedSegment = 1;
			continue;
		}
		if (segmentNumber != expectedSegment)
		{
			expectedSegment = 1;
			if (segmentNumber != 1)
			{
				continue;
			}
		}
		memcpy(shelf[segmentNumber - 1], result, sizeof(result));
		expectedSegment++;
		if (segmentNumber != lepton_vospi::kSegmentCount)
		{
			continue;
		}
		expectedSegment = 1;

		for (int i = 0; i < lepton_vospi::kSegmentCount; ++i)
		{
			ssize_t sent_bytes = sendto(sockfd, shelf[i], sizeof(shelf[i]), 0, (const struct sockaddr *)&servaddr, sizeof(servaddr));
			if (sent_bytes < 0)
			{
				std::cerr << "Send failed" << std::endl;
				close(sockfd);
				return -1;
			}
		}
		softResyncs = 0;
		usleep(10000);
	}
	// Close the socket
	close(sockfd);
	//finally, close SPI port just bcuz
	SpiClosePort(0);
}
