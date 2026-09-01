cmake_minimum_required(VERSION 3.22)
project(raspberrypi_video_network)

include(CTest)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/raspberrypi_video_network/include)
include_directories(${CMAKE_SOURCE_DIR}/raspberrypi_libs/leptonSDKEmb32PUB)

# Define the executable
add_executable(raspberrypi_video_network raspberrypi_video_network/src/main.cpp raspberrypi_video_network/src/SPI.cpp raspberrypi_video_network/src/Lepton_I2C.cpp)

# Ensure the library path is correct
link_directories(${CMAKE_SOURCE_DIR}/raspberrypi_libs/leptonSDKEmb32PUB/Debug)

# Custom commands to make sure the SDK is up to date
add_custom_target(sdk
    COMMAND make -C ${CMAKE_SOURCE_DIR}/raspberrypi_libs/leptonSDKEmb32PUB
    COMMENT "Building the SDK"
)

add_custom_target(sdkclean
    COMMAND make -C ${CMAKE_SOURCE_DIR}/raspberrypi_libs/leptonSDKEmb32PUB clean
    COMMENT "Cleaning the SDK"
)

# Ensure the SDK is built before the main project
add_dependencies(raspberrypi_video_network sdk)

# Add the SDK library
add_library(LEPTON_SDK STATIC IMPORTED)
set_target_properties(LEPTON_SDK PROPERTIES
    IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/raspberrypi_libs/leptonSDKEmb32PUB/Debug/libLEPTON_SDK.a
)

# Link the libraries
target_link_libraries(raspberrypi_video_network LEPTON_SDK)

if(BUILD_TESTING)
    add_executable(
        test_lepton_vospi
        raspberrypi_video_network/tests/test_lepton_vospi.cpp
    )
    add_test(NAME lepton_vospi_protocol COMMAND test_lepton_vospi)
endif()

# Set the output directories
set_target_properties(raspberrypi_video_network PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/.
    OBJECT_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/gen_objs
)
