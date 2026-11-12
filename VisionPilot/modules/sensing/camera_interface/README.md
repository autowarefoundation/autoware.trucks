# V4L2 Interface Module

## I. Overview

This **V4L2 Interface** module provides a robust C++ wrapper for capturing video frames from V4L2 (Video4Linux2) camera devices and converting them to OpenCV `cv::Mat` format. 

This module is essential for low-level camera frame acquisition in VisionPilot 1.0, featuring:

- V4L2 camera capturing whose interface directly connects to V4L2 mounted devices (`/dev/video0`, `/dev/video1`, etc.)
- Seamless conversion to OpenCV `cv::Mat` for downstream processing
- Mutex-protected thread-safe frame and statistics
- Able to specify desired FPS and codec settings
- Monitor frames captured, dropped, and errors, etc.

## II. Architecture & module structure

### 1. Architecture

The `V4L2Reader` class is the main interface for V4L2 camera operations. It follows a similar pattern to the ROS2 camera subscriber but operates at a lower hardware level.

Basically, for the V4L2 capture, we use a simple, off-the-shelf yet effective [OpenCV's VideoCapture](https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html) :

```cpp

cv::Mat frame;
camera_capture >> frame;  // Core OpenCV VideoCapture operation

```

This uses OpenCV's VideoCapture with the CAP_V4L2 backend for direct V4L2 device access.

```

┌─────────────────────────────────────────┐
│   Linux V4L2 Framework (/dev/videoX)    │
│   USB Video Device Driver               │
│   Camera Hardware                       │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│   OpenCV VideoCapture (CAP_V4L2)         │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│   V4L2 Interface Module                  │
│  ┌─────────────────────────────────────┐│
│  │  V4L2Reader Class                    ││
│  │  - get_latest_frame()                ││
│  │  - Statistics & monitoring           ││
│  └─────────────────────────────────────┘│
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│   VisionPilot Application Layer          │
└─────────────────────────────────────────┘

```

### 2. Module structure

```

v4l2_interface/
├── CMakeLists.txt
├── README.md
├── include/
│   └── v4l2_interface/
│       └── v4l2_reader.hpp
└── src/
    └── v4l2_reader.cpp

```

## III. Build

### 1. Prerequisites

- ROS2 (tested on ROS2 Humble / Ubuntu 22.04).
    - `source /opt/ros/humble/setup.bash`
- Required packages:
    - `OpenCV`

### 2. Steps

```bash

# 1. From root to release directory
cd VisionPilot/development_releases/1.0

# 2. Create build dir
mkdir -p build && cd build

# 3. Source ROS2 just in case you forgot to
source /opt/ros/humble/setup.bash

# 4. CMake configure
cmake ..

# 5. Compile
make -j$(nproc)

```

### 3. Expected output

```bash

[ 97%] Building CXX object app/CMakeFiles/VisionPilot.dir/vision_pilot.cpp.o
[100%] Linking CXX executable ../VisionPilot
[100%] Built target VisionPilot

```

## IV. Test

This module features V4L2 => OpenCV stream conversion, thus a V4L2 image stream is required.

You can do a simple test using a local video by simply just:

1. Publishing it as a V4L2 stream via [ffmpeg](https://ffmpeg.org/).

```bash

# 1. Install package
sudo apt update
sudo apt install ffmpeg -y
sudo apt install v4l2loopback-dkms -y

# 2. Load the module (assuming you gonna stream it at `/dev/video9`)
sudo modprobe -r v4l2loopback
sudo modprobe v4l2loopback video_nr=9 card_label="Virtual Camera" exclusive_caps=1

# 3. Publish looping video at that mount
ffmpeg -re -stream_loop -1 -i <absolute path to local video> -f v4l2 -pix_fmt yuv420p /dev/video9

```

With this the V4L2 stream at `dev/video9` is now active. You can have a look at it from another terminal using:

```bash

ffplay /dev/video9

```

2. Then, on another terminal (or back to the terminal that you built VisionPilot), we can test it inside the main VisionPilot loop:

```bash

# First argument `1` means starting this with V4L2 input
# Second argument being mounted V4L2 stream
# Third argument being desired FPS
./VisionPilot 1 /dev/video9 10

```