import math
import carla
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from sensor_msgs.msg import Image
from sensor_msgs.msg import Imu


class CarlaVehicleSpeedPublisherNode(Node):
    def __init__(self):
        super().__init__('carla_vehicle_speed_publisher_node')

        self.declare_parameter('host', 'localhost')
        self.declare_parameter('port', 2000)
        # self.image_sub_ = self.create_subscription(Image, '/carla/hero/main_cam/image', self.image_callback, 1)
        self.imu_sub_ = self.create_subscription(Imu, '/carla/hero/imu', self.imu_callback, 1)
        self.speed_pub_ = self.create_publisher(Float64, '/vehicle/speed', 1)

        host = self.get_parameter('host').get_parameter_value().string_value
        port = self.get_parameter('port').get_parameter_value().integer_value
        self.client = carla.Client(host, port)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()

        self.hero_actor = None
        self._try_find_hero_actor()

        # Retry until the ego vehicle actor is actually spawned/queryable.
        self.actor_search_timer = self.create_timer(0.5, self._retry_find_hero_actor)

    def _try_find_hero_actor(self):
        for actor in self.world.get_actors().filter('vehicle.*'):
            if actor.attributes.get('role_name') == 'hero':
                self.hero_actor = actor
                return True
        return False

    def _retry_find_hero_actor(self):
        if self._try_find_hero_actor():
            self.get_logger().info(f'hero_actor found (id={self.hero_actor.id}), ready.')
            self.actor_search_timer.cancel()

    # def image_callback(self, msg):
    #     if self.hero_actor is None:
    #         self.get_logger().warn('hero_actor not found yet, skipping frame', throttle_duration_sec=2.0)
    #         return
    #
    #     velocity = self.hero_actor.get_velocity()
    #     # speed_kmh = 3.6 * math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
    #     speed_kmh = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
    #
    #     out = Float64()
    #     out.data = speed_kmh
    #     self.speed_pub_.publish(out)
    #     self.get_logger().info(f'Velocity published: {speed_kmh:.2f} km/h')

    # def image_callback(self, msg):
    #     self.get_logger().info("img stamp=%d.%09d" % (msg.header.stamp.sec, msg.header.stamp.nanosec))

    def imu_callback(self, msg: Imu):
        if self.hero_actor is None:
            self.get_logger().warn('hero_actor not found yet, skipping', throttle_duration_sec=2.0)
            return

        # 2. Get local non-blocking snapshot matching the latest server tick (No world.tick()!)
        world_snapshot = self.world.get_snapshot()
        hero_snapshot = world_snapshot.find(self.hero_actor.id)

        if hero_snapshot:
            # 3. Retrieve exact velocity recorded for this frame
            velocity = hero_snapshot.get_velocity()
            speed_m_s = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

            # 4. Publish result
            out = Float64()
            out.data = speed_m_s
            self.speed_pub_.publish(out)
            # self.get_logger().info(f'Velocity published: {speed_m_s:.2f} m/s')


def main(args=None):
    rclpy.init(args=args)
    node = CarlaVehicleSpeedPublisherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()