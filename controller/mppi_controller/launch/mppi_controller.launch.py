import os
import launch
import launch_ros

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
import subprocess
from launch.actions import (OpaqueFunction, LogInfo, RegisterEventHandler)
from launch.event_handlers import (OnProcessExit)

# Exit process function
def exit_process_function(_launch_context):
    subprocess.run('ros2 topic pub -t 1 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"', shell=True)

# Generate launch description
def generate_launch_description():

    config_arg = DeclareLaunchArgument(
        'config',
        default_value='params_gazebo.yaml',
        description='Name of the configuration file to load'
    )
    
    # Ghép đường dẫn bằng Substitution ➜ an toàn ở runtime
    config_path = PathJoinSubstitution([
        get_package_share_directory('mppi_controller'),
        'config',
        LaunchConfiguration('config')
    ])


    # Node definition
    node = Node(
        package='mppi_controller',
        executable='mppi_controller_human_acc.py',
        output='screen',
        emulate_tty=True,
        parameters=[config_path]
    )

    return LaunchDescription([
        config_arg,
        node,
        RegisterEventHandler(
            OnProcessExit(
                target_action=node,
                on_exit=[
                    LogInfo(msg='MPPI exiting'),
                    LogInfo(msg='Stopping robot'),
                    OpaqueFunction(
                        function=exit_process_function
                    ),
                    LogInfo(msg='Stopping done'),
                ]
            )
        )
    ])
