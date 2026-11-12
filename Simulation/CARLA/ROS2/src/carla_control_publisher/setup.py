from setuptools import setup

package_name = 'carla_control_publisher'

setup(
    name=package_name,
    version='1.1.0',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Atanasko Boris Mitrev',
    maintainer_email='atanasko.mitrev@autoware.org',
    description='carla_control_publisher for CARLA simulation in ROS2',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'carla_control_publisher_node = carla_control_publisher.carla_control_publisher_node:main',
        ],
    },
)
