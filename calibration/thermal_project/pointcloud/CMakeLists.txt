cmake_minimum_required(VERSION 3.5)
project(ThermalPointCloud)

# Find PCL package
find_package(PCL 1.14 REQUIRED)

set(OpenGL_GL_PREFERENCE LEGACY)
find_package(OpenGL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(GLUT REQUIRED)
find_package(glfw3 REQUIRED)
find_package(realsense2 REQUIRED)

# Include headers
include_directories(${PCL_INCLUDE_DIRS} ${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS} ${GLFW_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/../stream/include)

# Link PCL libraries
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Define the executables
add_executable(thermalPC thermal_pc.cpp ../stream/Palettes.cpp)
add_executable(loadPC load_pc.cpp)

# Link the libraries
target_link_libraries(thermalPC ${DEPENDENCIES} ${PCL_LIBRARIES} ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} glfw ${realsense2_LIBRARY} ${OpenCV_LIBS})
target_link_libraries(loadPC ${DEPENDENCIES} ${PCL_LIBRARIES} ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} glfw ${realsense2_LIBRARY} ${OpenCV_LIBS})

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)