# Thermal Point Cloud

#### Author: Anuj Natraj

https://github.com/user-attachments/assets/0584e4b3-5929-4763-acd6-b2016e8b9847

## Package List

There are 3 main packages in this project each with their own builds. Apart from this you will need the Lepton 3.1R thermal camera and a RaspberryPi to stream the thermal data from the camera to your server.

I have made a [package](https://github.com/AnujN9/LeptonModule/tree/master/software/raspberrypi_video_network) that sends the data from the RaspberryPi via a UDP socket. 

- stream - streams images and can save the images
    - streams raw data from the Lepton and converts it to an image
    - streams data from a RealSense D435i and the Lepton
- calibration - performs the camera calibration from images saved in its directory
    - calibration of intrinsics for the thermal camera. Saves to a calibration.xml file.
    - calibration of the extrinsics between the thermal camera and the RealSense. Saves to an extrinsic.xml file.
- pointcloud - generates a point cloud from the depth data and thermal data
    - creates a point cloud where the depth data determines the position and the thermal data determines the color of points based on a colormap. Also capable of saving the point cloud and loading it seperately.

I am currently working on a [package](https://github.com/AnujN9/ThermalProject_ROS) port all this into ROS 2.

I have added a project.repos which contains all 3 repositories. ```vcs import --input https://raw.githubusercontent.com/AnujN9/ThermalProject/main/project.repos``` to download all at once into the directory.

## Project Flow

### Data

The data flow for this project is as follows:

![DataFlow](docs/assets/dataflow.png)

I set up the RaspberryPi with a Lepton 3.1R. The thermal camera communicates between the RPi using I2C and sends the images via SPI. Once this data is acquired using a simple program to read the SPI communication, I send this raw data via UDP through an ethernet cable to my Laptop. A RealSense D435i is connnected via USB to the laptop. The data from the thermal camera and the RealSense is processed into a thermal pointcloud.

### Work Flow

To set up a point cloud of your own, follow these steps:
- First connect up an RPi with a Lepton camera using this [package](https://github.com/AnujN9/LeptonModule/tree/master/software/raspberrypi_video_network) to send raw data. You can find a [3D printable case](docs/thermal_case.step) that can house an RPi, Lepton camera and a Realsense as one unit.
- Once it is set up and streams data from the ethernet, use ```./lepton``` to stream the images and save images for calibration. Collect more than 10 views so at least 10 complete detections remain, and place `thermal_grayimage_N.png` in `calibration/thermal_images`.
- The supplied [pattern](docs/chessboard_pattern.dxf) has 5x4 physical squares (30 mm) and therefore 4x3 OpenCV inner corners. Run ```./camera_calibration -r 4 -c 5 -n <image-count> -pat 1``` from the calibration build directory to find the intrinsics.
- Capture thermal/RGB pairs with ```./depth_saver```. Hold the board stationary while pressing `c`, then place `thermal_grayimage_N.png` in `calibration/thermal_images` and `color_image_N.png` in `calibration/color_images`. At least 10 pairs must have complete detections in both cameras.
- Run ```./extrinsic -r 4 -c 5 -n <pair-count>``` from the calibration build directory to find the transform of the thermal camera in the RGB frame.
- `pointcloud` is used to generate point clouds. Move the 2 .xml files from `calibration` to `pointcloud`. Use ```./thermalPC``` to generate a stream of point clouds. You can save a point cloud by pressing 's'. Use ```./loadPC``` to load that saved point cloud.


## References

- https://henryzh47.github.io/assets/documents/multiple-methods-geometric.pdf
- https://github.com/groupgets/LeptonModule
- https://github.com/IntelRealSense/librealsense/blob/development/wrappers/pcl/pcl/rs-pcl.cpp
