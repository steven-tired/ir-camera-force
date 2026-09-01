#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include "example.hpp" // Include short list of convenience functions for rendering

#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/io/pcd_io.h>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <ctime>
#include <cstdint>
#include <cstring>
#include <arpa/inet.h>
#include <Palettes.h>

#define PACKET_SIZE 164
#define PACKET_SIZE_UINT16 (PACKET_SIZE/2)
#define PACKETS_PER_FRAME 60
#define FRAME_SIZE_UINT16 (PACKET_SIZE_UINT16*PACKETS_PER_FRAME)
#define FPS 27

/// \file thermal_pc.cpp
/// \brief Program that streams a pointcloud combining depth points with thermal data. Can save a point cloud.

/// \brief 3D position state for displaying pointcloud
struct state
{
    state() : yaw(0.0), pitch(0.0), last_x(0.0), last_y(0.0),
        ml(false), offset_x(0.0f), offset_y(0.0f) {}
    double yaw, pitch, last_x, last_y;
    bool ml;
    float offset_x, offset_y;
};

using pcl_ptr = pcl::PointCloud<pcl::PointXYZRGB>::Ptr;

void register_glfw_callbacks(window& app, state& app_state);
void draw_pointcloud(window& app, state& app_state, const std::vector<pcl_ptr>& points);
void process_thermaldata(uint8_t (*shelf)[PACKET_SIZE * PACKETS_PER_FRAME], cv::Mat& gray, cv::Mat& color, int& myImageWidth, int& myImageHeight, uint16_t& rangeMin,  uint16_t& rangeMax);

/// \brief Converts depth points to pointcloud with rgb values based on thermal colormap.
/// \param points Depth points from the realsense depth frame.
/// \param thermalimage Thermal image that contains the temperature data.
/// \param cameraMatrixThermal Thermal camera intrinsics.
/// \param R Rotation matrix to convert depth camera frame to thermal camera frame.
/// \param T Translation matrix to convert depth camera frame to thermal camera frame.
/// \return PCL XZYRGB pointcloud.
pcl_ptr points_to_pcl(const rs2::points& points, cv::Mat& thermalimage, const cv::Mat& cameraMatrixThermal, const cv::Mat& R, const cv::Mat& T)
{
    pcl_ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    auto sp = points.get_profile().as<rs2::video_stream_profile>();
    cloud->width = sp.width();
    cloud->height = sp.height();
    cloud->is_dense = false;
    cloud->points.resize(points.size());
    auto ptr = points.get_vertices();
    for (auto& p : cloud->points)
    {
        p.x = ptr->x;
        p.y = ptr->y;
        p.z = ptr->z;
        ptr++;

        if (p.z > 0)
        {
            cv::Mat realSensePoint(3, 1, CV_64F);
            realSensePoint.at<double>(0, 0) = p.x;
            realSensePoint.at<double>(1, 0) = p.y;
            realSensePoint.at<double>(2, 0) = p.z;
            cv::Mat thermalPoint = R * realSensePoint + T;
            cv::Mat projectedThermalPoint = cameraMatrixThermal * thermalPoint;
            int thermalX = static_cast<int>(projectedThermalPoint.at<double>(0, 0) / projectedThermalPoint.at<double>(2, 0));
            int thermalY = static_cast<int>(projectedThermalPoint.at<double>(1, 0) / projectedThermalPoint.at<double>(2, 0));

            if (thermalX >= 0 && thermalX < thermalimage.cols && thermalY >= 0 && thermalY < thermalimage.rows)
            {
                cv::Vec3b color = thermalimage.at<cv::Vec3b>(thermalY, thermalX);
                p.r = color[2];
                p.g = color[1];
                p.b = color[0];
            }
            else
            {
                p.r = 153;
                p.g = 153;
                p.b = 153;
            }
        }
    }
    return cloud;
}

/// \brief Function to describe how to use the command line arguments
/// \param cmd Argument of the command line, here it is the program
void printUsage(char *cmd)
{
	char *cmdname = basename(cmd);
	printf(" Usage: %s [OPTION]...\n"
		   " -h			display this help and exit.\n"
		   " -port x		set the ip port (default: 8080).\n"
		   " -mintemp x		sets a minimum value for scaling (suggestion: 27300).\n"
		   " -maxtemp x		sets a maximum value for scaling (suggestion: 30800).\n"
		   "			Temperature values for min and max are in hectoKelvin.\n"
		   " Output:		Pointcloud stream where the rgb values are a temperature map.\n"
		   "", cmdname);
	return;
}

/// \brief Creates and streams pointcloud.
/// \param argc Number of command-line arguments.
/// \param argv Array of command-line arguments.
/// \param port Port of the IP address.
/// \param mintemp Minimum temperature to be scaled between 0 and 255.
/// \param maxtemp Maximum temperature to be scaled between 0 and 255.
/// \return 0 if successful, -1 if failure.
int main(int argc, char * argv[])
try
{
    window app(1280, 720, "RealSense PCL Pointcloud Example");
    state app_state;
    register_glfw_callbacks(app, app_state);

    uint16_t rangeMin = 27300;
	uint16_t rangeMax = 30800;
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
				rangeMin = static_cast<uint16_t>(temp);
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
				rangeMax = static_cast<uint16_t>(temp);
			}
			else
			{
				std::cerr << "Error: Enter a maximum temp (0 to 65535)." << std::endl;
				exit(1);
			}
		}
	}

    int sockfd;
    struct sockaddr_in servaddr, cliaddr;

    uint8_t shelf[4][PACKET_SIZE * PACKETS_PER_FRAME];
    bool first = true;
    int myImageWidth = 160;
    int myImageHeight = 120;
    cv::Mat color(myImageHeight, myImageWidth, CV_8UC3);
    cv::Mat gray(myImageHeight, myImageWidth, CV_8UC1);
    cv::Mat undistortedColor(myImageHeight, myImageWidth, CV_8UC3);
    cv::Mat undistortedImage(myImageHeight, myImageWidth, CV_8UC1);
    
    if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0)
    {
        std::cerr << "Socket creation failed" << std::endl;
        return -1;
    }
    memset(&servaddr, 0, sizeof(servaddr));
    memset(&cliaddr, 0, sizeof(cliaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(port);
    servaddr.sin_addr.s_addr = INADDR_ANY;
    if (bind(sockfd, (const struct sockaddr *)&servaddr, sizeof(servaddr)) < 0)
    {
        std::cerr << "Bind failed" << std::endl;
        close(sockfd);
        return -1;
    }
    socklen_t len = sizeof(cliaddr);

    cv::FileStorage fs("../calibration.xml", cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        std::cerr << "Failed to open calibration.xml" << std::endl;
        return -1;
    }
    cv::Mat cameraMatrixThermal, distCoeffsThermal;
    fs["cameraMatrix"] >> cameraMatrixThermal;
    fs["distCoeffs"] >> distCoeffsThermal;
    fs.release();
    cv::FileStorage fs2("../extrinsic.xml", cv::FileStorage::READ);
    if (!fs2.isOpened())
    {
        std::cerr << "Failed to open extrinsic.xml" << std::endl;
        return -1;
    }
    cv::Mat R_rgb_thermal, T_rgb_thermal, E, F, R_thermal_rgb, T_thermal_rgb;
    fs2["R"] >> R_rgb_thermal;
    fs2["T"] >> T_rgb_thermal;
    fs2["E"] >> E;
    fs2["F"] >> F;
    fs2.release();
    R_thermal_rgb = R_rgb_thermal.inv();
    T_thermal_rgb = -R_thermal_rgb * T_rgb_thermal;

    rs2::pointcloud pc;
    rs2::points points;
    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_Z16);
    cfg.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_RGB8);
    pipe.start(cfg);
    rs2::align align_to_color(RS2_STREAM_COLOR);

    while (app)
    {
        auto frames = pipe.wait_for_frames();
        frames = align_to_color.process(frames);
        auto depth = frames.get_depth_frame();
        for (int i = 0; i < 4; ++i)
        {
            ssize_t received_bytes = recvfrom(sockfd, shelf[i], sizeof(shelf[i]), 0, (struct sockaddr *)&cliaddr, &len);
            if (received_bytes < 0)
            {
                std::cerr << "Receive failed" << std::endl;
                close(sockfd);
                return -1;
            }
            if (first && i == 1)
            {
                break;
            }
        }
        if (first)
        {
            first = false;
            continue;
        }
        process_thermaldata(shelf, gray, color, myImageWidth, myImageHeight, rangeMin, rangeMax);
        cv::undistort(gray, undistortedImage, cameraMatrixThermal, distCoeffsThermal);
        cv::undistort(color, undistortedColor, cameraMatrixThermal, distCoeffsThermal);
        points = pc.calculate(depth);
        auto pcl_points = points_to_pcl(points, undistortedColor, cameraMatrixThermal, R_thermal_rgb, T_thermal_rgb);

        // pcl_ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
        // pcl::PassThrough<pcl::PointXYZRGB> pass;
        // pass.setInputCloud(pcl_points);
        // pass.setFilterFieldName("z");
        // pass.setFilterLimits(0.0, 1.0);
        // pass.filter(*cloud_filtered);

        std::vector<pcl_ptr> layers;
        layers.push_back(pcl_points);
        // layers.push_back(cloud_filtered);

        // cv::imshow("Thermal", undistortedImage);
        cv::imshow("Thermal Color", undistortedColor);
        int key = cv::waitKey(1);
        draw_pointcloud(app, app_state, layers);

        if (key == 's')
        {
            pcl::io::savePCDFileASCII ("thermal.pcd", *pcl_points);
            std::cout << "Saved pointcloud" << std::endl;
        }
    }
    close(sockfd);
    return EXIT_SUCCESS;
}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception & e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}

/// \brief Function to create callbacks to interact with the point cloud.
/// \param app Window that renders the point cloud. Interactions happen due to the callbacks here.
/// \param app_state State of the app's current transform.
/// \return None.
void register_glfw_callbacks(window& app, state& app_state)
{
    app.on_left_mouse = [&](bool pressed)
    {
        app_state.ml = pressed;
    };

    app.on_mouse_scroll = [&](double xoffset, double yoffset)
    {
        app_state.offset_x += static_cast<float>(xoffset);
        app_state.offset_y += static_cast<float>(yoffset);
    };

    app.on_mouse_move = [&](double x, double y)
    {
        if (app_state.ml)
        {
            app_state.yaw -= (x - app_state.last_x);
            app_state.yaw = std::max(app_state.yaw, -120.0);
            app_state.yaw = std::min(app_state.yaw, +120.0);
            app_state.pitch += (y - app_state.last_y);
            app_state.pitch = std::max(app_state.pitch, -80.0);
            app_state.pitch = std::min(app_state.pitch, +80.0);
        }
        app_state.last_x = x;
        app_state.last_y = y;
    };

    app.on_key_release = [&](int key)
    {
        if (key == 32)
        { // Escape
            app_state.yaw = app_state.pitch = 0;
            app_state.offset_x = app_state.offset_y = 0.0;
        }
    };
}

/// \brief OpenGL function to render the point cloud with rgb and xyz.
/// \param app Window that renders the point cloud.
/// \param app_state State than contains the view transforms.
/// \param points Point cloud vector to be rendered.
/// \return None.
void draw_pointcloud(window& app, state& app_state, const std::vector<pcl_ptr>& points)
{
    glPopMatrix();
    glPushAttrib(GL_ALL_ATTRIB_BITS);

    float width = app.width(), height = app.height();

    glClearColor(153.f / 255, 153.f / 255, 153.f / 255, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    gluPerspective(60, width / height, 0.01f, 10.0f);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    gluLookAt(0, 0, 0, 0, 0, 1, 0, -1, 0);

    glTranslatef(0, 0, +0.5f + app_state.offset_y * 0.05f);
    glRotated(app_state.pitch, 1, 0, 0);
    glRotated(app_state.yaw, 0, 1, 0);
    glTranslatef(0, 0, -0.5f);

    glPointSize(width / 640);
    glEnable(GL_TEXTURE_2D);

    for (auto&& pc : points)
    {
        glBegin(GL_POINTS);
        for (int i = 0; i < pc->points.size(); i++)
        {
            auto&& p = pc->points[i];
            if (p.z && p.z < 2.0)
            {
                glColor3ub(p.r, p.g, p.b);
                glVertex3f(p.x, p.y, p.z);
            }
        }
        glEnd();
    }

    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glPopAttrib();
    glPushMatrix();
}

/// \brief Processes data from UDP to thermal image.
/// \param shelf Raw temperature image data.
/// \param gray Thermal image that is generated.
/// \param myImageWidth Width of the image.
/// \param myImageHeight Height of the image.
/// \param rangeMin Minimum temperature to be scaled between 0 and 255.
/// \param rangeMax Maximum temperature to be scaled between 0 and 255.
/// \return None.
void process_thermaldata(uint8_t (*shelf)[PACKET_SIZE * PACKETS_PER_FRAME],
                        cv::Mat& gray,
                        cv::Mat& color,
                        int& myImageWidth,
                        int& myImageHeight,
                        uint16_t& rangeMin,
                        uint16_t& rangeMax)
{
    float diff = rangeMax - rangeMin;
    float scale = 255 / diff;
    uint16_t n_zero_value_drop_frame = 0;
    const int *selectedColormap = colormap_ironblack;
	int selectedColormapSize = get_size_colormap_ironblack();

    int row, column;
    uint16_t value;
    uint16_t valueFrameBuffer;
    for (int iSegment = 1; iSegment <= 4; iSegment++)
    {
        int ofsRow = 30 * (iSegment - 1);
        for (int i = 0; i < FRAME_SIZE_UINT16; i++)
        {
            if (i % PACKET_SIZE_UINT16 < 2)
            {
                continue;
            }
            valueFrameBuffer = (shelf[iSegment - 1][i * 2] << 8) + shelf[iSegment - 1][i * 2 + 1];
            if (valueFrameBuffer == 0)
            {
                n_zero_value_drop_frame++;
                break;
            }
            if (valueFrameBuffer <= rangeMin)
            {
                value = 0;
            }
            else if (valueFrameBuffer > rangeMax)
            {
                value = 255;
            }
            else
            {
                value = ((valueFrameBuffer - rangeMin) * scale);
            }
            int ofs_r = 3 * value + 0; if (selectedColormapSize <= ofs_r) ofs_r = selectedColormapSize - 1;
            int ofs_g = 3 * value + 1; if (selectedColormapSize <= ofs_g) ofs_g = selectedColormapSize - 1;
            int ofs_b = 3 * value + 2; if (selectedColormapSize <= ofs_b) ofs_b = selectedColormapSize - 1;
            cv::Vec3b rgbcolor(selectedColormap[ofs_b], selectedColormap[ofs_g], selectedColormap[ofs_r]);
            column = (i % PACKET_SIZE_UINT16) - 2 + (myImageWidth / 2) * ((i % (PACKET_SIZE_UINT16 * 2)) / PACKET_SIZE_UINT16);
            row = i / PACKET_SIZE_UINT16 / 2 + ofsRow;
            if (row < myImageHeight && column < myImageWidth)
            {
                color.at<cv::Vec3b>(row, column) = rgbcolor;
                gray.at<uint8_t>(row, column) = value;
            }
        }
    }
    if (n_zero_value_drop_frame != 0)
    {
        n_zero_value_drop_frame = 0;
    }
}
