cmake_minimum_required(VERSION 3.22.1)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

add_library(vp_debug
    src/debug_draw.cpp
)

target_include_directories(vp_debug
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include   # <debug/...>
)

target_link_libraries(vp_debug
    PUBLIC
        models          # AutoDriveOutput, AutoSteerOutput, AutoSpeedOutput
        fusion          # CIPOFusionEstimate, LateralFusionEstimate
        opencv_core
        opencv_imgproc
        opencv_imgcodecs
)

target_compile_features(vp_debug PUBLIC cxx_std_17)
