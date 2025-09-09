import os
import launch
import launch_ros

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import subprocess
from launch.actions import (OpaqueFunction, LogInfo, RegisterEventHandler)
from launch.event_handlers import (OnProcessExit)

# Generate launch description
def generate_launch_description():
    # Path to params.yaml
    config = os.path.join(
        get_package_share_directory('vlm_strategy'),
        'config',
        'params_experiment.yaml'
    )

    # Node definition
    node = Node(
        package='vlm_strategy',
        executable='vlm_strategy.py',
        output='screen',
        emulate_tty=True,
        parameters=[config]
    )

    return LaunchDescription([
        node
    ])
