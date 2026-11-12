from setuptools import setup
import os
from glob import glob

package_name = 'carla_bridge_bringup'

setup(
    name=package_name,
    version='1.1.0',
    packages=[],
    data_files=[
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*_launch.py')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'scripts'), glob('scripts/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Atanasko Boris Mitrev',
    maintainer_email='atanasko.mitrev@autoware.org',
    description='Vision Pilot bringup launch files',
    license='Apache License 2.0',
)
