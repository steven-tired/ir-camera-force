This example is meant for Raspberry Pi 4 and Lepton 3.1R.

## Installation

First enable the SPI and I2C interfaces on the Pi.
```
sudo raspi-config
```

Install cmake:
```
sudo apt install cmake -y
```

Clone the repo.

To build, cd to the *LeptonModule/sofware* folder, then run:
```
mkdir build
cd build
cmake ..
make
```

----

### Lepton 3.x

Before running ensure the RPi is running on performance mode. Modify the */etc/rc.local*, add the command at the end and reboot:

```
sudo sh -c "echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"
```

To run the program while still in the build directory, using a FLIR Lepton 3.x camera core use the following code:
```
./raspberrypi_video_network
```
It has command arguments to set the IP address and port

----

I made a [package](https://github.com/AnujN9/ThermalProject) that acts as the other end of the bridge and has a few more capabilties. The purpose of the package was to combine the depth data from a RealSense D435i and the thermal data from the Lepton 3.1R to create a thermal point cloud.

----
In order for the application to run properly, a Lepton camera must be attached through the camera breakout board V2.0 in a specific way to the SPI, power, and ground pins of the Raspi's GPIO interface, as well as the I2C SDA/SCL pins:

Please follow the [Getting started with the Raspberry Pi and Breakout Board V2.0.pdf](https://github.com/AnujN9/LeptonModule/tree/master/docs) document in the `LeptonModule/docs` folder