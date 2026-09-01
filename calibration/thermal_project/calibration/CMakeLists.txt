cmake_minimum_required(VERSION 3.10)
project(Calibration)

include(CTest)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
find_package(OpenCV REQUIRED)
find_package(realsense2 REQUIRED)

# Include directories for OpenCV
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Define the Image directory for calibration
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/thermal_images)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/color_images)
add_definitions(-DTHERMAL_IMAGE_DIRECTORY="${CMAKE_SOURCE_DIR}/thermal_images")
add_definitions(-DIMAGE_DIRECTORY="${CMAKE_SOURCE_DIR}/color_images")

# Define the executable
add_library(calibration_contracts CalibrationContracts.cpp)
target_link_libraries(calibration_contracts PUBLIC ${OpenCV_LIBS})

add_library(heldout_verifier HeldOutVerifier.cpp)
target_link_libraries(heldout_verifier PUBLIC calibration_contracts ${OpenCV_LIBS})

add_executable(camera_calibration cal.cpp)
add_executable(verify_calibration verify.cpp)
add_executable(extrinsic extrinsic_cal.cpp)
add_executable(resolve_extrinsic resolve_extrinsic.cpp)
add_executable(heldout_verify heldout_verify.cpp)

# Link the libraries to executables
target_link_libraries(camera_calibration calibration_contracts ${OpenCV_LIBS})
target_link_libraries(verify_calibration ${OpenCV_LIBS})
target_link_libraries(extrinsic heldout_verifier calibration_contracts ${OpenCV_LIBS} ${realsense2_LIBRARY})
target_link_libraries(resolve_extrinsic calibration_contracts ${OpenCV_LIBS} ${realsense2_LIBRARY})
target_link_libraries(heldout_verify heldout_verifier calibration_contracts
    ${OpenCV_LIBS} ${realsense2_LIBRARY})

if(BUILD_TESTING)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)

    add_executable(calibration_contract_test
        tests/calibration_contract_test.cpp)
    target_include_directories(calibration_contract_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(calibration_contract_test calibration_contracts ${OpenCV_LIBS})
    add_test(NAME calibration_contract_test COMMAND calibration_contract_test)

    add_executable(heldout_verifier_test tests/heldout_verifier_test.cpp)
    target_include_directories(heldout_verifier_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_definitions(heldout_verifier_test PRIVATE
        HELDOUT_REPORT_JSON_PATH="${CMAKE_CURRENT_BINARY_DIR}/heldout_report.json")
    target_link_libraries(heldout_verifier_test heldout_verifier calibration_contracts ${OpenCV_LIBS})
    add_test(NAME heldout_verifier_test COMMAND heldout_verifier_test)
    add_test(NAME heldout_report_stdlib_json_test
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/validate_report_json.py
            ${CMAKE_CURRENT_BINARY_DIR}/heldout_report.json)
    set_tests_properties(heldout_report_stdlib_json_test PROPERTIES
        DEPENDS heldout_verifier_test)
    add_test(NAME heldout_verify_help_test
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/validate_help.py
            $<TARGET_FILE:heldout_verify>)
endif()
