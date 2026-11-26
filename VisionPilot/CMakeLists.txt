cmake_minimum_required(VERSION 3.22.1)

project(VisionPilot VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Put all executables in build/
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Testing (default OFF; users can override with -DBUILD_TESTING=ON)
set(BUILD_TESTING OFF CACHE BOOL "Build the testing tree")

# GPU support
option(GPU "Enable GPU support" ON)

# ROS2 packages
option(ENABLE_ROS2_INTERFACE "Enable ROS2 support" OFF)

# Build testing code
option(BUILD_TESTING "Build the testing tree" OFF)

# Rerun C++ SDK
# Disabled by default
# Resolved either from a local checkout pointed to by RERUN_SDK_ROOT (env or cache var),
# or downloaded via FetchContent.
option(ENABLE_RERUN "Log directly to a Rerun .rrd via the Rerun C++ SDK" OFF)

set(Python3_EXECUTABLE /usr/bin/python3 CACHE FILEPATH "" FORCE)
find_package(Python3 REQUIRED)

# Optional directory whose files replace the defaults in config/ (e.g. a simulator-specific
# vision_pilot.conf, vision_pilot_ros2.conf and H.yaml). The default config/ is left untouched.
set(VISIONPILOT_CONFIG_OVERLAY "" CACHE PATH "Directory with config files overriding config/")
if(VISIONPILOT_CONFIG_OVERLAY)
    get_filename_component(VISIONPILOT_CONFIG_OVERLAY "${VISIONPILOT_CONFIG_OVERLAY}" ABSOLUTE)
    if(NOT IS_DIRECTORY "${VISIONPILOT_CONFIG_OVERLAY}")
        message(FATAL_ERROR "VISIONPILOT_CONFIG_OVERLAY is not a directory: ${VISIONPILOT_CONFIG_OVERLAY}")
    endif()
    message(STATUS "VisionPilot config overlay: ${VISIONPILOT_CONFIG_OVERLAY}")
endif()

if(VISIONPILOT_CONFIG_OVERLAY AND EXISTS "${VISIONPILOT_CONFIG_OVERLAY}/H.yaml")
    set(VISIONPILOT_GROUND_H "${VISIONPILOT_CONFIG_OVERLAY}/H.yaml")
else()
    set(VISIONPILOT_GROUND_H "${CMAKE_SOURCE_DIR}/config/H.yaml")
endif()

add_custom_target(generate_config ALL
        COMMAND Python3::Interpreter
        ${CMAKE_SOURCE_DIR}/scripts/find_homography_C_matrix.py
        --ground-h ${VISIONPILOT_GROUND_H}
        --output ${CMAKE_BINARY_DIR}/config/homography_C_matrix.yaml
)

# Modules
add_subdirectory(modules/logging)

add_subdirectory(modules/config)

add_subdirectory(modules/common)

add_subdirectory(modules/sensing/image_preprocessing)

if(ENABLE_ROS2_INTERFACE)
    add_subdirectory(modules/middleware_interfaces/ros2_interface/camera_ros2_interface)
    add_subdirectory(modules/middleware_interfaces/ros2_interface/vehicle_ros2_interface)
endif()

add_subdirectory(modules/engine)
add_subdirectory(modules/models)

add_subdirectory(modules/safety_guardian/fusion)   # tracking is embedded inside fusion
add_subdirectory(modules/safety_guardian/planning)

add_subdirectory(modules/sensing/vehicle_interface)
add_subdirectory(modules/sensing/camera_interface)

add_subdirectory(modules/visualization)

add_subdirectory(modules/debug)

# Tests
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

# config
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/config
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
if(VISIONPILOT_CONFIG_OVERLAY)
    file(GLOB VISIONPILOT_OVERLAY_FILES "${VISIONPILOT_CONFIG_OVERLAY}/*")
    file(COPY ${VISIONPILOT_OVERLAY_FILES}
            DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/config)
endif()

# App
add_subdirectory(app)

set(CPACK_PACKAGE_VENDOR       "Autoware Foundation")
set(CPACK_PACKAGE_CONTACT      "zain.khawaja@autoware.org, atanasko.mitrev@autoware.org, pranav.doma@autoware.org, tran.huy@autoware.org")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "${CPACK_PACKAGE_CONTACT}")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "VisionPilot 1.0")

set(CPACK_DEBIAN_PACKAGE_DEPENDS
        "libopencv-dev,
        libgstreamer1.0-0,
        libgstreamer-plugins-base1.0-0,
        libgstreamer-plugins-bad1.0-0,
        libnice10,
        gstreamer1.0-nice,
        libsrtp2-1,
        coinor-libipopt-dev,
        liblapack3,
        libblas3"
)

if(GPU)
    set(APPEND CPACK_DEBIAN_PACKAGE_DEPENDS
        "cuda-toolkit-13-3,
        libcudnn9-cuda-13,
        libcublas-13-3,
        cuda-cudart-13-3,
        libnvinfer10,
        libnvinfer-plugin10,
        libnvonnxparsers10,
        nvidia-open"
    )
endif()

# Architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(ARCH_NAME "x86_64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(ARCH_NAME "arm64")
else()
    set(ARCH_NAME "${CMAKE_SYSTEM_PROCESSOR}")
endif()

# GPU/CPU support
if(GPU)
    set(PKG_SUFFIX "_gpu")
else()
    set(PKG_SUFFIX "_cpu")
endif()

set(CPACK_PACKAGE_NAME "VisionPilot")
set(CPACK_SYSTEM_NAME "${ARCH_NAME}${PKG_SUFFIX}")
set(CPACK_GENERATOR   "DEB")

include(CPack)
