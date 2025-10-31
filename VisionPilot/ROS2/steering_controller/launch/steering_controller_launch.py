from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='steering_controller',  
            executable='steering_controller_node',
            name='steering_controller_node',
            output='screen'
        ),
        
        Node(
            package='longitudinal_controller',  
            executable='longitudinal_controller_node',
            name='longitudinal_controller_node',
            output='screen'
        ),
        
        Node(
            package='carla_control_publisher',
            executable='pub_carla_control',
            name='pub_carla_control',
            output='screen'
        ),
        
        Node(
            package='road_shape_publisher',
            executable='road_shape_publisher_node',
            name='road_shape_publisher_node',
            output='screen'
        ),
        
        Node(
            package='PATHFINDER',
            executable='pathfinder_node',
            name='pathfinder_node',
            output='screen'
        ),
        
        Node(
            package='odom_publisher',
            executable='pub_odom_node',
            name='pub_odom_node',
            output='screen'
        ),
        
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='front_to_base_broadcaster',
            arguments=[
                "1.425", "0", "0",   # translation: x y z
                "0", "0", "0",   # rotation in rpy (roll pitch yaw in radians)
                "hero",           # parent frame
                "hero_front"           # child frame
            ]
        ),

    ])