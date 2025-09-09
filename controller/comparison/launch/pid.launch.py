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

# Exit process function
def exit_process_function(_launch_context):
    subprocess.run('ros2 topic pub -t 1 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"', shell=True)

# Generate launch description
def generate_launch_description():
    # Path to params.yaml
    config = os.path.join(
        get_package_share_directory('comparison'),
        'config',
        'params.yaml'
    )

    # Node definition
    node = Node(
        package='comparison',
        executable='PID.py',
        output='screen',
        emulate_tty=True,
        parameters=[config]
    )

    return LaunchDescription([
        node,
        RegisterEventHandler(
            OnProcessExit(
                target_action=node,
                on_exit=[
                    LogInfo(msg='PID exiting'),
                    LogInfo(msg='Stopping robot'),
                    OpaqueFunction(
                        function=exit_process_function
                    ),
                    LogInfo(msg='Stopping done'),
                ]
            )
        )
    ])
