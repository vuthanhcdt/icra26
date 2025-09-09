import os, launch_ros
from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('amfitrack'),
        'config',
        'params.yaml'
        )

    node = launch_ros.actions.Node(
        package='amfitrack',
        executable='amfitrack.py',
        output='screen',
        emulate_tty=True,
        parameters=[config])
    
    return LaunchDescription([
        node
    ])
