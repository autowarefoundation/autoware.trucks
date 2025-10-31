# deploy_libtorch
This application uses libTorch C++ to load model and inference.

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

Configure project:

```
cmake -DLIBTORCH_INSTALL_ROOT=<_libTorch_root_location_> -DOPENCV_INSTALL_ROOT=<_opencv_root_location_> ..
```

Build the project:

```
make
```

Run Network:

```
./deploy_libtorch <_input_network_file.pt_> <_input_image_file.png_>
```

If the application runs successfully it will produce two output files:

1) Segmentation mask image file.
2) Output image file consisting of segmentation mask overlayed onto the input image.