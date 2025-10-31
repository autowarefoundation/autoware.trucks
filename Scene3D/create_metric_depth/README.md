
## Scene3D Metric Depth Dataset
To train the Scene3D metric depth network, a custom dataset was created using a number of open-source, publicly available datasets which capture scene depth through LIDAR scanners. Raw depth maps were interpolated to fill in holes and create a densified depth ground-truth. Validity maps were created to account for areas of missing data which would not be penalized during network training.

The Scene3D Metric Depth Dataset comprises RGB images with associated ground truth depth maps and validity masks indicating pixels where valid depth measurements are present.

![Scene3D_Metric_Data](../../Diagrams/Scene3D_Metric_Data.jpg)

The datasets used to train SuperDepth include:

- [Argoverse](https://www.argoverse.org/av1.html#stereo-link)
- [KITTI](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction)
- [DDAD](https://github.com/TRI-ML/DDAD#dataset-details)
- [DrivingStereo](https://drivingstereo-dataset.github.io/)

## common
Contains helper class to perform LIDAR depth map interpolation and densification