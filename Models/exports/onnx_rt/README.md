# deploy_onnx_rt
This application uses ONNX Runtime to load model and inference on the following ONNX Runtime Execution Providers:

1) CUDA
2) DNNL (oneDNN)

## Build Instructions
Set up environment (if using CUDA):

```
export PATH=/usr/local/cuda-12.3/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

Create build location:

```
mkdir build && cd build

```

To configure project with CUDA Execution Provider (EP):

```

cmake   -DLIBTORCH_INSTALL_ROOT=<_libTorch_root_location_> 
        -DOPENCV_INSTALL_ROOT=<_opencv_root_location_> 
        -DONNXRUNTIME_ROOTDIR=<_onnx_runtime_root_location_>
        -DUSE_CUDA_BACKEND=True
        ..

```

To configure project with DNNL Execution Provider (EP):

```
cmake   -DLIBTORCH_INSTALL_ROOT=<_libTorch_root_location_> 
        -DOPENCV_INSTALL_ROOT=<_opencv_root_location_> 
        -DONNXRUNTIME_ROOTDIR=<_onnx_runtime_root_location_>
        -DUSE_DNNL_BACKEND=True
        ..

```

Build Project:

```
make
```

Run Network:

```
./deploy_onnx_rt <_input_network_file.onnx_> <_input_image_file.png_>
```

If the application runs successfully it will produce two output files:

1) Output segmentation mask image file.
2) Output image file consisting of segmentation mask overlayed onto the input image.