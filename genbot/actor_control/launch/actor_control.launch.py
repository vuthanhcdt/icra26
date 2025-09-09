import launch_ros
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Cho phép người dùng chọn file YAML
    config_arg = DeclareLaunchArgument(
        'config',
        default_value='params.yaml',
        description='Name of the configuration file to load'
    )

    # Ghép đường dẫn bằng Substitution ➜ an toàn ở runtime
    config_path = PathJoinSubstitution([
        get_package_share_directory('actor_control'),
        'config',
        LaunchConfiguration('config')
    ])

    static_tf = launch_ros.actions.Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_designed_point',
        arguments=[
            '0.3', '0.9', '0.0',          # x y z
            '0', '0', '0',                # roll pitch yaw (rad)
            'human_local_link', 'designed_point'
        ],
        output='screen'
    )

    node = launch_ros.actions.Node(
        package='actor_control',
        executable='actor_control.py',
        parameters=[config_path],        # <‑‑ dùng file YAML vừa ghép
        output='screen',
        emulate_tty=True
    )

    return LaunchDescription([
        config_arg,
        static_tf,
        node
    ])
