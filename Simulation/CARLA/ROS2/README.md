# VisionPilot ⇄ CARLA (0.9.16) — ROS 2

Closed-loop simulation of VisionPilot in CARLA. VisionPilot reads the camera image and the vehicle
speed from CARLA and drives the ego vehicle with its steering and acceleration commands.

No Autoware and no CARLA ROS bridge are needed: the camera comes from CARLA's native ROS 2
publisher (`--ros2`), and speed/control go through the CARLA PythonAPI.

```text
CARLA (--ros2) ──/carla/hero/main_cam/image─────────────────────────────▶ VisionPilot
CARLA PythonAPI ──carla_vehicle_speed_publisher──/vehicle/speed────────▶ VisionPilot
VisionPilot ──/vehicle/steering_cmd, /vehicle/throttle_cmd──▶ carla_control_publisher ──PythonAPI──▶ CARLA ego
```

- [Quick start: PoV Highway truck demo](#quick-start-pov-highway-truck-demo)
  - [Prerequisites](#prerequisites)
  - [Setup (once)](#setup-once)
  - [Run](#run)
  - [Stop](#stop)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Known issues](#known-issues)

## Quick start: PoV Highway truck demo

Drives the `vehicle.carlamotors.european_hgv` truck on the `PoV_Highway` map.

### Prerequisites

| Requirement                                                                     | Notes                                                                                                                                                                                                                                                                 |
| ------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Ubuntu 22.04 + [ROS 2 Humble](https://docs.ros.org/en/humble/Installation.html) | `ros-humble-desktop` or `ros-humble-ros-base`                                                                                                                                                                                                                         |
| NVIDIA GPU + driver, CUDA 12 runtime libraries + cuDNN 9                        | For CARLA rendering and VisionPilot GPU inference. ONNX Runtime needs cuDART, cuBLAS, cuRAND, **cuFFT** and cuDNN 9, e.g. `sudo apt install cuda-libraries-12-8 libcudnn9-cuda-12` from the [NVIDIA CUDA apt repository](https://developer.nvidia.com/cuda-downloads) |
| CARLA 0.9.16 **with the `PoV_Highway` map**                                     | Not part of the stock CARLA release. Download the package linked in [autoware.trucks#4](https://github.com/autowarefoundation/autoware.trucks/issues/4#issuecomment-5521889863) and extract it                                                                        |
| CARLA PythonAPI 0.9.16                                                          | `pip install carla==0.9.16` (must match the server version)                                                                                                                                                                                                           |
| ONNX Runtime GPU (CUDA 12)                                                      | Installed from NuGet with [`VisionPilot/scripts/install_onnxruntime.sh`](../../../VisionPilot/scripts/install_onnxruntime.sh), see [Setup](#2-install-onnx-runtime). The GitHub release tarballs are no longer available                                              |
| OpenCV development files                                                        | `sudo apt install libopencv-dev`                                                                                                                                                                                                                                      |

The commands below use these variables. Adjust them to your environment:

```bash
export AUTOWARE_TRUCKS=~/autoware.trucks                                # this repository
export CARLA_ROOT=~/CARLA_Shipping_0.9.16/LinuxNoEditor                 # directory containing CarlaUE4.sh
export ONNXRUNTIME_ROOT=/opt/onnx_runtime/onnxruntime-linux-x64-gpu_cuda12-1.26.0  # see Setup step 2
```

Check that the map is present:

```bash
ls $CARLA_ROOT/CarlaUE4/Content/Carla/Maps/PoV_Highway.umap
```

### Setup (once)

#### 1. Install the CARLA PythonAPI

```bash
pip install carla==0.9.16
python3 -c "import carla; print(carla.__file__)"
```

#### 2. Install ONNX Runtime

The ONNX Runtime GitHub release tarballs are no longer available, so install it from the NuGet
package `Microsoft.ML.OnnxRuntime.Gpu.Linux`. The script downloads it from nuget.org and lays it
out like the former tarball (`include/` + `lib/`):

```bash
sudo $AUTOWARE_TRUCKS/VisionPilot/scripts/install_onnxruntime.sh 1.26.0 /opt/onnx_runtime
# -> ONNX Runtime 1.26.0 installed at /opt/onnx_runtime/onnxruntime-linux-x64-gpu_cuda12-1.26.0
```

Use **1.26.0 or older**: from 1.27.0 the NuGet package is built for CUDA 13, and the script
refuses it. Any directory you can write to works as the second argument (then `sudo` is not
needed); set `ONNXRUNTIME_ROOT` to the path the script prints. The script needs `curl`, `unzip`
and `readelf` (`binutils`).

#### 3. Build the ROS 2 bridge nodes

```bash
cd $AUTOWARE_TRUCKS/Simulation/CARLA/ROS2
source /opt/ros/humble/setup.bash
colcon build
```

#### 4. Build VisionPilot for CARLA

```bash
cd $AUTOWARE_TRUCKS/VisionPilot
source /opt/ros/humble/setup.bash
mkdir -p build && cd build
cmake .. \
  -DENABLE_ROS2_INTERFACE=ON \
  -DONNXRUNTIME_ROOT=$ONNXRUNTIME_ROOT \
  -DVISIONPILOT_CONFIG_OVERLAY=$AUTOWARE_TRUCKS/Simulation/CARLA/ROS2/config/vision_pilot
make -j"$(nproc)"
```

`-DVISIONPILOT_CONFIG_OVERLAY` puts the CARLA settings from
[`config/vision_pilot/`](config/vision_pilot) into `build/config/`. It sets ROS 2 input mode, the
CARLA camera topic, and the camera homography `H.yaml` for the truck camera. The defaults in
`VisionPilot/config/` are not changed.

Check that the overlay was applied:

```bash
grep -E "source.mode|input_camera_topic" config/vision_pilot.conf config/vision_pilot_ros2.conf
# config/vision_pilot.conf:source.mode             = ros2
# config/vision_pilot_ros2.conf:source.input_camera_topic   = /carla/hero/main_cam/image
```

### Run

Use three terminals, in this order.

> [!IMPORTANT]
> CARLA's native ROS 2 publisher always uses DDS domain 0. Set `ROS_DOMAIN_ID=0` in every ROS 2
> terminal. With any other domain the topics show up, but no data arrives.

#### Terminal 1: CARLA server

```bash
$CARLA_ROOT/CarlaUE4.sh --ros2
```

Wait until the CARLA window shows the default map (takes a few seconds to a minute).

#### Terminal 2: map, truck and bridge nodes

```bash
cd $AUTOWARE_TRUCKS/Simulation/CARLA/ROS2
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=0
ros2 launch carla_bridge_bringup carla_bridge.launch.py rig_file:=$(pwd)/config/pov_highway.json
```

This loads `PoV_Highway`, spawns the truck on road 2 / lane -2 / s = 500 m, runs the simulation in
synchronous mode at 10 Hz and moves the CARLA spectator camera behind the truck. It is ready when
the log shows:

```text
[carla_control_publisher]: hero actor found (id=...), ready.
[carla_vehicle_speed_publisher]: hero_actor found (id=...), ready.
```

The first load of `PoV_Highway` takes longer (`No InMemoryMap cache found...` is expected).

#### Terminal 3: VisionPilot

```bash
cd $AUTOWARE_TRUCKS/VisionPilot/build
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0
./VisionPilot
```

VisionPilot must be started from `build/`, because it reads `config/` and the model weights relative
to the current directory. A `VisionPilot` window opens and the truck starts to accelerate and keep
its lane. The terminal prints one `plan:` line per frame.

#### Checking the data flow (optional)

In another terminal:

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0
ros2 topic hz /carla/hero/main_cam/image   # ~10 Hz
ros2 topic echo /vehicle/speed             # m/s
ros2 topic echo /vehicle/steering_cmd      # published by VisionPilot
```

### Stop

Press `Ctrl+C` in terminal 3, then terminal 2, then close CARLA. Stopping terminal 2 restores
CARLA to asynchronous mode and removes the truck.

## Configuration

### Rig files

A rig file describes the world the bridge sets up:

| File                                                 | Map           | Vehicle                                    | Camera                                           |
| ---------------------------------------------------- | ------------- | ------------------------------------------ | ------------------------------------------------ |
| [`config/pov_highway.json`](config/pov_highway.json) | `PoV_Highway` | `vehicle.carlamotors.european_hgv` (truck) | 1080x720, fov 120, cab front (x 3.85 m, z 2.7 m) |
| [`config/carla916.json`](config/carla916.json)       | `Town04`      | `vehicle.lincoln.mkz_2020`                 | 1920x1280, fov 50                                |

| Key              | Description                                                                                                                       |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `map`            | CARLA map to load (default `Town04`)                                                                                              |
| `type`, `id`     | Ego blueprint and its `role_name` / ROS name. Keep `id` = `hero` (the topic names and nodes use it)                               |
| `spawn_waypoint` | Spawn at an OpenDRIVE location: `road_id`, `lane_id`, `s` [m], optional `z_offset` [m] (default 1.0)                              |
| `spawn_index`    | Used when `spawn_waypoint` is absent: index into the map's spawn points. The `SPAWN_INDEX` environment variable overrides it      |
| `sensors`        | Sensors attached to the ego. `spawn_point` uses ROS conventions (x forward, y left, z up; degrees) relative to the vehicle origin |
| `npc_vehicles`   | Optional traffic (`type`, `count`, `spawn_index`, `autopilot`)                                                                    |

Pass a rig file with `rig_file:=<absolute path>`.

### VisionPilot settings for CARLA

[`config/vision_pilot/`](config/vision_pilot) is the overlay applied by
`-DVISIONPILOT_CONFIG_OVERLAY`:

| File                     | Content                                                                                                              |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------- |
| `vision_pilot.conf`      | Same as `VisionPilot/config/vision_pilot.conf`, with `source.mode = ros2`                                            |
| `vision_pilot_ros2.conf` | Camera topic `/carla/hero/main_cam/image` and vehicle topics                                                         |
| `H.yaml`                 | Ground homography for the `pov_highway.json` camera. Recalibrate it if you change the camera pose, resolution or fov |

The overlay is copied when `cmake` runs. After editing it, run `cmake` again before `make`.

### Packages

| Package                         | Role                                                                                                                                                                    |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `carla_bridge_bringup`          | `config_carla.py` loads the map, spawns the ego and sensors, ticks the simulation and moves the spectator. `carla_bridge.launch.py` starts it, then the two nodes below |
| `carla_vehicle_speed_publisher` | Publishes the ego speed [m/s] on `/vehicle/speed`                                                                                                                       |
| `carla_control_publisher`       | Applies `/vehicle/steering_cmd` [rad, positive = right] and `/vehicle/throttle_cmd` [m/s²] to the ego as `carla.VehicleAckermannControl`                                |

## Troubleshooting

| Symptom                                                                                                                                                      | Cause / fix                                                                                                                                                                                                                                   |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ros2 topic list` shows `/carla/hero/...` but `ros2 topic hz` gets nothing, and the truck does not move                                                      | `ROS_DOMAIN_ID` is not 0 in the bridge or VisionPilot terminal. `export ROS_DOMAIN_ID=0`                                                                                                                                                      |
| `config_carla.py` warns `CARLA PythonAPI ... != server ...` or crashes                                                                                       | PythonAPI version does not match the server. `pip install carla==0.9.16`                                                                                                                                                                      |
| Terminal 2 fails right after `Loading map PoV_Highway`                                                                                                       | The CARLA package does not contain the map (check with the `ls` command in [Prerequisites](#prerequisites)). Use the package from [autoware.trucks#4](https://github.com/autowarefoundation/autoware.trucks/issues/4#issuecomment-5521889863) |
| VisionPilot: `Can't initialize GTK backend`                                                                                                                  | No display available (for example over SSH). Run it on the desktop, or use `xvfb-run -a ./VisionPilot`                                                                                                                                        |
| VisionPilot: `Config file not found` / model not found                                                                                                       | VisionPilot was not started from `VisionPilot/build/`                                                                                                                                                                                         |
| VisionPilot: `ONNXRUNTIME_ROOT is not set` at `cmake`                                                                                                        | Pass `-DONNXRUNTIME_ROOT=<install_dir>/onnxruntime-linux-x64-gpu_cuda12-<version>` (printed by `install_onnxruntime.sh`)                                                                                                                      |
| VisionPilot: `Failed to load library .../libonnxruntime_providers_cuda.so with error: libcufft.so.11: cannot open shared object file` (or another `lib*.so`) | A CUDA 12 runtime library is missing. Install it, e.g. `sudo apt install libcufft-12-8` (or the whole `cuda-libraries-12-8`)                                                                                                                  |
| `install_onnxruntime.sh`: `is not a CUDA 12 build`                                                                                                           | NuGet ships CUDA 13 builds from ONNX Runtime 1.27.0. Use 1.26.0 or older                                                                                                                                                                      |
| VisionPilot reads `/camera/image` instead of `/carla/hero/main_cam/image`                                                                                    | Built without `-DVISIONPILOT_CONFIG_OVERLAY`. Re-run `cmake` with it                                                                                                                                                                          |

## Known issues

- On `PoV_Highway` the truck hits an overhead road sign at about s = 1774 m on road 2, roughly
  1.3 km after the spawn point, and gets pushed out of its lane. See
  [autoware.trucks#4](https://github.com/autowarefoundation/autoware.trucks/issues/4).
