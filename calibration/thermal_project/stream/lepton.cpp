#include <iostream>
#include <ctime>
#include <stdint.h>
#include <thread>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <arpa/inet.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <array>
#include <vector>
#include <ThermalFrameAssembler.h>
#include <Palettes.h>

#define PACKET_SIZE 164
#define PACKET_SIZE_UINT16 (PACKET_SIZE / 2)
#define PACKETS_PER_FRAME 60
#define FRAME_SIZE_UINT16 (PACKET_SIZE_UINT16 * PACKETS_PER_FRAME)
#define FPS 27;

/// \file depthimage.cpp
/// \brief Program that streams thermal images and saves it to thermal_images directory.

/// \brief Function to describe how to use the command line arguments
/// \param cmd Argument of the command line, here it is the program
void printUsage(char *cmd)
{
	char *cmdname = basename(cmd);
	printf(" Usage: %s [OPTION]...\n"
		   " -h			display this help and exit.\n"
		   " -port x		set the ip port (default: 8080).\n"
		   " -mintemp x		sets a minimum value for scaling (suggestion: 27300).\n"
		   " -maxtemp x		sets a maximum value for scaling (suggestion: 33500).\n"
		   "			Temperature values for min and max are in hectoKelvin.\n"
		   " Capture:		To capture images press c on the image window.\n"
		   "			Saves raw grayscale and custom colormap images\n"
		   "			to the thermal_images directory.\n"
		   "", cmdname);
	return;
}

/// \brief Converts the raw data to a thermal image and streams the rgb image of the realsense.
/// \param argc Number of command-line arguments.
/// \param argv Array of command-line arguments.
/// \param port Port of the IP address.
/// \param mintemp Minimum temperature to be scaled between 0 and 255.
/// \param maxtemp Maximum temperature to be scaled between 0 and 255.
/// \return 0 if successful, -1 if failure.
int main(int argc, char **argv)
{
	// Socket initialization
    int sockfd;
    struct sockaddr_in servaddr, cliaddr;

	// Lepton image setting
    const int *selectedColormap = colormap_ironblack;
	int selectedColormapSize = get_size_colormap_ironblack();
	bool autoRangeMin = true;
	bool autoRangeMax = true;
	uint16_t rangeMin = 27300; // Minimum temperture range (temp / 100 - 273) celsius
	uint16_t rangeMax = 33500; // Maximum temperture range
	int myImageWidth = 160;
	int myImageHeight = 120;
	int img_cnt = 0;

	uint16_t port = 8080;

	for(int i=1; i < argc; i++)
	{
		if (strcmp(argv[i], "-h") == 0)
		{
			printUsage(argv[0]);
			exit(0);
		}
		else if (strcmp(argv[i], "-port") == 0)
		{
			if (i + 1 != argc)
			{
				long int temp = std::strtol(argv[++i], nullptr, 10);
				if (temp < 0 || temp > 65535){
					std::cerr << "Error: Enter a valid Port." << std::endl;
					exit(1);
				}
				port = static_cast<uint16_t>(temp);
			}
			else
			{
				std::cerr << "Error: Enter a valid Port." << std::endl;
				exit(1);
			}
		}
		else if (strcmp(argv[i], "-mintemp") == 0)
		{
			if (i + 1 != argc)
			{
				long int temp = std::strtol(argv[++i], nullptr, 10);
				if (temp < 0 || temp > 65535){
					std::cerr << "Error: Enter a temp between 0 and 65525." << std::endl;
					exit(1);
				}
				rangeMin = temp;
				autoRangeMin = false;
			}
			else
			{
				std::cerr << "Error: Enter a minimum temp (0 to 65535)." << std::endl;
				exit(1);
			}
		}
		else if (strcmp(argv[i], "-maxtemp") == 0)
		{
			if (i + 1 != argc)
			{
				long int temp = std::strtol(argv[++i], nullptr, 10);
				if (temp < 0 || temp > 65535){
					std::cerr << "Error: Enter a temp between 0 and 65525." << std::endl;
					exit(1);
				}
				rangeMax = temp;
				autoRangeMax = false;
			}
			else
			{
				std::cerr << "Error: Enter a maximum temp (0 to 65535)." << std::endl;
				exit(1);
			}
		}
	}

	thermal_stream::ThermalFrameAssembler thermalAssembler;
	std::array<uint8_t, thermal_stream::ThermalFrameAssembler::kDatagramBytes + 1> datagram;
	std::vector<uint16_t> thermalFrame;
	uint16_t minValue = rangeMin;
	uint16_t maxValue = rangeMax;
	float diff = maxValue - minValue;
	float scale = 255 / diff;
	uint16_t n_zero_value_drop_frame = 0;

    cv::Mat image(myImageHeight, myImageWidth, CV_8UC3);
	cv::Mat gray(myImageHeight, myImageWidth, CV_8UC1);

    // Create socket
    if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0)
	{
        std::cerr << "Socket creation failed" << std::endl;
        return -1;
    }

    memset(&servaddr, 0, sizeof(servaddr));
    memset(&cliaddr, 0, sizeof(cliaddr));
    
    // Set up server address
    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(port);
    servaddr.sin_addr.s_addr = INADDR_ANY;

    // Bind the socket
    if (bind(sockfd, (const struct sockaddr *)&servaddr, sizeof(servaddr)) < 0)
	{
        std::cerr << "Bind failed" << std::endl;
        close(sockfd);
        return -1;
    }
    socklen_t len = sizeof(cliaddr);

    while (true)
	{
        bool thermalFrameComplete = false;
        while (!thermalFrameComplete)
		{
            ssize_t received_bytes = recvfrom(sockfd, datagram.data(), datagram.size(), 0, (struct sockaddr *)&cliaddr, &len);
            if (received_bytes < 0)
			{
                std::cerr << "Receive failed" << std::endl;
                close(sockfd);
                return -1;
            }
			thermalFrameComplete = thermalAssembler.add_datagram(
				datagram.data(), static_cast<std::size_t>(received_bytes), thermalFrame) ==
				thermal_stream::ThermalFrameAssembler::Result::Complete;
        }

		if (autoRangeMin || autoRangeMax)
		{
			if (autoRangeMin)
			{
				maxValue = 65535;
			}
			if (autoRangeMax)
			{
				maxValue = 0;
			}
			for (std::size_t i = 0; i < thermalFrame.size(); ++i)
			{
				uint16_t value = thermalFrame[i];
				if (value == 0)
				{
					continue;
				}
				if (autoRangeMax && (value > maxValue))
				{
					maxValue = value;
				}
				if (autoRangeMin && (value < minValue))
				{
					minValue = value;
				}
			}
			diff = maxValue - minValue;
			scale = 255 / diff;
		}

		int row, column;
		uint16_t value;
		uint16_t valueFrameBuffer;
		const auto renderRanges = thermal_stream::renderable_pixel_ranges(thermalFrame);
		for (std::size_t segment = 0; segment < renderRanges.size(); ++segment)
		{
			const auto &range = renderRanges[segment];
			const std::size_t segmentEnd = segment + 1 < renderRanges.size()
				? renderRanges[segment + 1].begin
				: thermalFrame.size();
			if (range.end < segmentEnd)
			{
				n_zero_value_drop_frame++;
			}
			for (std::size_t i = range.begin; i < range.end; ++i)
			{
				valueFrameBuffer = thermalFrame[i];

				if (!autoRangeMin && (valueFrameBuffer <= minValue))
				{
					value = 0;
				}
				else if (!autoRangeMax && (valueFrameBuffer > maxValue))
				{
					value = 255;
				}
				else
				{
					value = ((valueFrameBuffer - minValue) * scale);
				}
				int ofs_r = 3 * value + 0; if (selectedColormapSize <= ofs_r) ofs_r = selectedColormapSize - 1;
				int ofs_g = 3 * value + 1; if (selectedColormapSize <= ofs_g) ofs_g = selectedColormapSize - 1;
				int ofs_b = 3 * value + 2; if (selectedColormapSize <= ofs_b) ofs_b = selectedColormapSize - 1;
				cv::Vec3b color(selectedColormap[ofs_b], selectedColormap[ofs_g], selectedColormap[ofs_r]);
                column = i % myImageWidth;
                row = i / myImageWidth;
                image.at<cv::Vec3b>(row, column) = color;
				gray.at<uint8_t>(row, column) = value;
                if ((column == 80) && (row == 60))
				{
                    double temp = (valueFrameBuffer / 100) - 273;
                    std::cout << "Temp at center " << temp << std::endl;
                }
			}
		}

		if (n_zero_value_drop_frame != 0)
		{
			n_zero_value_drop_frame = 0;
		}
		cv::imshow("Thermal Image", image);
		int k = cv::waitKey(1);
        if (k == 'c')
		{
			// Save the images when 'c' is pressed
			img_cnt++;
            std::string filename = "thermal_images/thermal_image_" + std::to_string(img_cnt) + ".png";
			cv::imwrite(filename, image);
			filename = "thermal_images/thermal_grayimage_" + std::to_string(img_cnt) + ".png";
			cv::imwrite(filename, gray);
			std::cout << "Image saved" << std::endl;
        }
		else if (k >= 0)
		{
			break;
		}
    }

    // Close the socket
    close(sockfd);
    return 0;
}