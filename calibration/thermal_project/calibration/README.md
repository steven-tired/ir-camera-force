# Calibration

### Installation

Run the following command to build this package to calibrate the Lepton 3.1R and then to calibrate the extrinsics between the Lepton 3.1R the RealSense D435i

```
mkdir build && cd build
cmake ..
make
```
Creates 3 executables, one for the thermal camera intrinsic calibration, one for verification of that calibration and one for extrinsic calibration between the thermal and RGB camera

### Usage

To calibrate the thermal camera, first ensure there are images in the `thermal_images` directory. When running you need to pass in parameters. Then run

```
./camera_calibration
    -r x    number of rows in the pattern
    -c x    number of columns in the pattern
    -n x    number of images
    -pat x  pattern type (1: chessboard, 2: symmetric dot)
```

For the supplied checkerboard, put at least 10 successfully detected
`thermal_grayimage_N.png` views in `thermal_images`, vary the board pose, and
run:

```
./camera_calibration -r 4 -c 5 -n <image-count> -pat 1
```

For this checkerboard, `-r 4 -c 5` describes 4x5 physical squares at 30 mm
spacing; OpenCV detects 4x3 inner corners.

This saves the intrinsics to a calibration.xml file. To see the undistorted images run

```
./verify_calibration
    -n x    number of images
```

To calibrate the extrinsics, put matching `thermal_grayimage_N.png` files in `thermal_images` and `color_image_N.png` files in `color_images`. While capturing with `depth_saver`, hold the board stationary while pressing `c`. At least 10 pairs must have complete 4x3 inner-corner detections in both cameras. Then run

```
./extrinsic -r 4 -c 5 -n <pair-count>
```
This will save the extrinsics between the thermal and rgb camera into an extrinsic.xml file.
