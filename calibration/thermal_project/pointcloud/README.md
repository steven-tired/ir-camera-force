# PointCloud

### Installation

NEEDED PCL 1.14

Install [pcl 1.14](https://github.com/PointCloudLibrary/pcl/releases) using this [guide](https://pcl.readthedocs.io/projects/tutorials/en/latest/compiling_pcl_posix.html) first.

Run the following command to build this package to create a PointCloud using the depth image from a RealSense D435i and project the thermal data from a Lepton 3.1R

```
mkdir build && cd build
cmake ..
make
```

### Usage

To stream the point cloud and the thermal image with a custom colormap, run

```
./thermalPC
    -port x         the port to recieve thermal data from
    -mintemp x      the minimum temperature used for scaling
    -maxtemp x      the maximum temperature used for scaling
```

Save the point cloud by pressing 's' on the image window and it will save it to thermal.pcd in the build directory

To load that point cloud, run

```
./loadPC
```
