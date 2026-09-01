cmake_minimum_required(VERSION 3.10)
project(lepton_project)

include(CTest)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Packages to run the programs
find_package(realsense2 REQUIRED)
find_package(OpenCV REQUIRED)

# Include directories for OpenCV and the current directory
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)

# Make directories for the images to be stored
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/thermal_images)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/images)

# Define the executable
add_executable(lepton lepton.cpp Palettes.cpp ThermalFrameAssembler.cpp)
add_executable(depth_saver depthimage.cpp Palettes.cpp ThermalFrameAssembler.cpp
    RealSenseCaptureContract.cpp)

if(BUILD_TESTING)
    add_executable(thermal_frame_assembler_test
        tests/thermal_frame_assembler_test.cpp
        ThermalFrameAssembler.cpp)
    target_include_directories(thermal_frame_assembler_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME thermal_frame_assembler_test COMMAND thermal_frame_assembler_test)

    add_executable(stream_contract_test
        tests/stream_contract_test.cpp
        RealSenseCaptureContract.cpp)
    target_include_directories(stream_contract_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_definitions(stream_contract_test PRIVATE
        DEPTHIMAGE_SOURCE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/depthimage.cpp")
    add_test(NAME stream_contract_test COMMAND stream_contract_test)
endif()

# Link the libraries
target_link_libraries(lepton ${OpenCV_LIBS})
target_link_libraries(depth_saver ${realsense2_LIBRARY} ${OpenCV_LIBS})
