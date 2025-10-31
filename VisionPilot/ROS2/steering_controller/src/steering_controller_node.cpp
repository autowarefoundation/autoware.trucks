#include "steering_controller_node.hpp"

SteeringControllerNode::SteeringControllerNode(const rclcpp::NodeOptions &options) : Node("steering_controller_node", "", options),
                                                                                    sc(2.85, 3.0, 0.1)
{
  sub_ = this->create_subscription<std_msgs::msg::Float32MultiArray>("/pathfinder/tracked_states", 10, std::bind(&SteeringControllerNode::stateCallback, this, std::placeholders::_1));
  odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>("/hero/odom", 10, std::bind(&SteeringControllerNode::odomCallback, this, std::placeholders::_1));
  steering_cmd_pub_ = this->create_publisher<std_msgs::msg::Float32>("/vehicle/steering_cmd", 1);
  RCLCPP_INFO(this->get_logger(), "SteeringController Node started");
  cte_ = 0.0;
  curvature_ = 0.0;
  forward_velocity_ = 0.0;
  steering_angle_ = 0.0;
  integral_error_ = 0.0;
  yaw_error_ = 0.0;
}

void SteeringControllerNode::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
  forward_velocity_ = msg->twist.twist.linear.x; // in m/s
}

void SteeringControllerNode::stateCallback(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
{
  if (msg->data.size() < 13)
  {
    RCLCPP_WARN(this->get_logger(), "Received message with insufficient data size: %zu", msg->data.size());
    return;
  }
  cte_ = msg->data[3];
  yaw_error_ = msg->data[7];
  curvature_ = msg->data[11];
  steering_angle_ = sc.computeSteering(cte_, yaw_error_, forward_velocity_, curvature_);
  auto steering_angle_msg = std_msgs::msg::Float32();
  steering_angle_msg.data = steering_angle_;
  steering_cmd_pub_->publish(steering_angle_msg);
}

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SteeringControllerNode>(rclcpp::NodeOptions()));
  rclcpp::shutdown();
  return 0;
}