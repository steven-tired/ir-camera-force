# Streaming

### Installation

Run the following command to build this package to stream images from the Lepton 3.1R or to stream both thermal images from the Lepton and color images from the RealSense D435i

```
mkdir build && cd build
cmake ..
make
```
Creates 2 executables and 2 directories that stored captured images from the stream

### Usage

To stream data from the lepton run

```
./lepton
```

To stream data from the lepton and the d435i run

```
./depth_saver
```

Both programs have the following commandline functionality:
```
-port x         the port to recieve thermal data from
-mintemp x      the minimum temperature used for scaling
-maxtemp x      the maximum temperature used for scaling
```

To save images while running the programs, press 'c' on the image window and it will save it to its respective directory in the build directory

### Lepton 3.1R Stream

To stream data from a Lepton 3.1R please use this [codebase](https://github.com/AnujN9/LeptonModule) and use the [raspberrypi_video_network](https://github.com/AnujN9/LeptonModule/tree/master/software/raspberrypi_video_network)
