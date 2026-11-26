"""Apply VisionPilot steering/acceleration commands to the CARLA ego vehicle.

Subscribes to VisionPilot's /vehicle/steering_cmd and /vehicle/throttle_cmd and applies them
to the ego (role_name = `role_name` parameter) through the CARLA PythonAPI as
carla.VehicleAckermannControl. No CARLA ROS messages (carla_msgs, ackermann_msgs) are needed.
"""

import carla
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

MAX_ACCELERATION = 1.5  # m/s²
MAX_DECELERATION = 3.0  # m/s²
A_DES_MIN = -8.0  # reject planner spikes below this (phantom CIPO gave -50)
A_DES_MAX = MAX_ACCELERATION


class CarlaControlPublisher(Node):
    def __init__(self):
        super().__init__("carla_control_publisher")

        self.declare_parameter("host", "localhost")
        self.declare_parameter("port", 2000)
        self.declare_parameter("role_name", "hero")

        host = self.get_parameter("host").get_parameter_value().string_value
        port = self.get_parameter("port").get_parameter_value().integer_value
        self.role_name = self.get_parameter("role_name").get_parameter_value().string_value

        self.client = carla.Client(host, port)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()
        self.hero_actor = None
        if not self._try_find_hero_actor():
            # Retry until the ego vehicle actor is actually spawned/queryable.
            self.actor_search_timer = self.create_timer(0.5, self._retry_find_hero_actor)

        self.steering_sub_ = self.create_subscription(
            Float64, "/vehicle/steering_cmd", self.steering_callback, 1
        )
        self.throttle_sub_ = self.create_subscription(
            Float64, "/vehicle/throttle_cmd", self.throttle_callback, 1
        )
        self.speed_sub_ = self.create_subscription(
            Float64, "/vehicle/speed", self.speed_callback, 1
        )

        self.speed = 0.0
        self.steering_angle_cmd = 0.0
        self.acceleration = 0.0

        # Publish only once BOTH a fresh steering and a fresh throttle have
        # arrived. Each planning cycle in VisionPilot emits one steering + one
        # throttle together, so this fires once per cycle, at the planner's own
        # rate, with no zero-order-hold delay.
        self.have_steering = False
        self.have_throttle = False

    def _try_find_hero_actor(self):
        for actor in self.world.get_actors().filter("vehicle.*"):
            if actor.attributes.get("role_name") == self.role_name:
                self.hero_actor = actor
                self.get_logger().info(f"{self.role_name} actor found (id={actor.id}), ready.")
                return True
        return False

    def _retry_find_hero_actor(self):
        if self._try_find_hero_actor():
            self.actor_search_timer.cancel()

    def publish_control(self):
        if self.hero_actor is None or not self.hero_actor.is_alive:
            self.get_logger().warn(
                f"{self.role_name} actor not available, dropping command", throttle_duration_sec=2.0
            )
            return

        v = max(0.0, self.speed + self.acceleration * 2.0)

        # VisionPilot uses CARLA's steering convention (radians, positive = right).
        control = carla.VehicleAckermannControl(
            steer=float(self.steering_angle_cmd),
            steer_speed=0.0,
            speed=float(v),
            acceleration=float(self.acceleration),
            jerk=10.0,
        )
        self.hero_actor.apply_ackermann_control(control)

    def try_publish(self):
        if self.have_steering and self.have_throttle:
            self.publish_control()
            self.have_steering = False
            self.have_throttle = False

    def steering_callback(self, msg):
        self.steering_angle_cmd = msg.data
        self.have_steering = True
        self.try_publish()

    def throttle_callback(self, msg):
        self.acceleration = msg.data
        self.have_throttle = True
        self.try_publish()

    def speed_callback(self, msg):
        self.speed = msg.data


def main(args=None):
    rclpy.init(args=args)
    node = CarlaControlPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
