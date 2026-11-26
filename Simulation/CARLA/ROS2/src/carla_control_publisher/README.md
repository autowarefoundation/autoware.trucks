# **CarlaControlPublisher Node**

## **Overview**

The `CarlaControlPublisher` node merges VisionPilot's steering and longitudinal commands and applies them to the
ego vehicle in the **CARLA Simulator** through the CARLA PythonAPI.

## **Output**

Commands are applied to the ego vehicle (actor with `role_name` = `role_name` parameter) through the
CARLA PythonAPI as `carla.VehicleAckermannControl` (steering angle, target speed, acceleration).
No CARLA ROS messages are published.

## **Subscribed Topics**

| Topic                   | Message Type           | Description                                                                   |
| ----------------------- | ---------------------- | ----------------------------------------------------------------------------- |
| `/vehicle/steering_cmd` | `std_msgs/msg/Float64` | Receives desired tire steering angle in radians (positive = right)            |
| `/vehicle/throttle_cmd` | `std_msgs/msg/Float64` | Receives desired acceleration in m/s². Negative is to reduce speed by braking |
| `/vehicle/speed`        | `std_msgs/msg/Float64` | Current ego speed in m/s, used to derive the target speed                     |

## **Parameters**

| Name        | Type   | Default     | Description                               |
| ----------- | ------ | ----------- | ----------------------------------------- |
| `host`      | string | `localhost` | CARLA server host                         |
| `port`      | int    | `2000`      | CARLA server port                         |
| `role_name` | string | `hero`      | `role_name` of the ego vehicle to control |

## **Example Usage**

### **Run the Node**

```bash
ros2 run carla_control_publisher carla_control_publisher_node
```
