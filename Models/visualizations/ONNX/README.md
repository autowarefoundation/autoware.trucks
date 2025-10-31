# ONNX Runtime C++ Visualizations for Scene Segmentation

This project contains C++ applications that use ONNX Runtime to perform inference with a Scene Segmentation model and generate visualizations.

Two applications are built:
-   `deploy_onnx_rt` (`main.cpp`): Processes a single input image and saves a visualized output image.
-   `video_visualization` (`video_visualization.cpp`): Processes an input video file and saves a new video with the segmentation results overlaid.

## Dependencies
- **OpenCV**: For image and video processing.
- **ONNX Runtime**: For model inference.
- **LibTorch**: Required *only* for the `deploy_onnx_rt` (single image) tool for its tensor manipulation capabilities.

---

## Build Instructions

### 1. Set up Environment (Optional: for CUDA)
If you plan to use the CUDA Execution Provider, ensure your environment variables are set correctly.
```bash
export PATH=/usr/local/cuda-12.3/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

### 2. Create Build Directory
From the `Onnx` directory:
```bash
mkdir build && cd build
```

### 3. Configure with CMake
You need to provide the root paths for your installations of LibTorch, OpenCV, and ONNX Runtime.

**To build with CPU support (DNNL):**
```bash
cmake .. \\
    -DLIBTORCH_INSTALL_ROOT=/path/to/libtorch \\
    -DOPENCV_INSTALL_ROOT=/path/to/opencv \\
    -DONNXRUNTIME_ROOTDIR=/path/to/onnxruntime \\
    -DUSE_DNNL_BACKEND=True
```

**To build with GPU support (CUDA):**
```bash
cmake .. \\
    -DLIBTORCH_INSTALL_ROOT=/path/to/libtorch \\
    -DOPENCV_INSTALL_ROOT=/path/to/opencv \\
    -DONNXRUNTIME_ROOTDIR=/path/to/onnxruntime \\
    -DUSE_CUDA_BACKEND=True
```

### 4. Build the Project
```bash
make
```

---

## Usage
After a successful build, you will find two executables in the `build` directory.

### Image Visualization
Processes a single image and produces two output image files.

**Command:**
```bash
./deploy_onnx_rt <path_to_model.onnx> <path_to_input_image.png>
```
**Output:**
-   `output_seg_mask.jpg`: The pure segmentation mask.
-   `output_image.jpg`: The input image with the segmentation mask overlaid.


### Video Visualization
Processes a video file and produces a new video file with segmentation overlaid.

**Command:**
```bash
./video_visualization <path_to_model.onnx> <path_to_input_video.mp4>
```
**Output:**
-   A new video file named `<input_video_name>_seg.avi` (e.g., `my_video_seg.avi`) in the build directory.