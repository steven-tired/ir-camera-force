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

/// \file load_pc.cpp
/// \brief Program that loads a saved point cloud in the form of thermal.pcd and renders it.

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

/// \brief Loads and renders a point cloud file 'thermal.pcd'.
/// \param argc Number of command-line arguments. No arguments.
/// \param argv Array of command-line arguments. No arguments.
/// \return 0 if successful, -1 if failure.
int main(int argc, char * argv[])
try
{
    window app(1280, 720, "RealSense PCL Pointcloud Example");
    state app_state;
    register_glfw_callbacks(app, app_state);

    pcl_ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    
    pcl::io::loadPCDFile ("thermal.pcd", *cloud);
    std::vector<pcl_ptr> layers;
    layers.push_back(cloud);

    while (app)
    {
        draw_pointcloud(app, app_state, layers);
    }
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

