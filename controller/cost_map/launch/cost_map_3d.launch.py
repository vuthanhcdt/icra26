import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable, GroupAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    # Directories
    bringup_dir = get_package_share_directory('nav2_bringup')
    autonoumous_navigation_dir = get_package_share_directory('cost_map')

    # Launch configurations
    namespace = LaunchConfiguration('namespace')
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    autostart = LaunchConfiguration('autostart', default='true')
    use_respawn = LaunchConfiguration('use_respawn', default='false')
    log_level = LaunchConfiguration('log_level', default='info')

    # Parameters
    param_file_name = 'params.yaml'
    params_file = os.path.join(autonoumous_navigation_dir, 'config', param_file_name)
    param_substitutions = {'autostart': autostart}
    configured_params = RewrittenYaml(
        source_file=params_file,
        root_key=namespace,
        param_rewrites=param_substitutions,
        convert_types=True,
    )

    # Environment variable
    stdout_linebuf_envvar = SetEnvironmentVariable('RCUTILS_LOGGING_BUFFERED_STREAM', '1')

    # Nodes
    load_nodes = GroupAction(
        actions=[
            Node(
                package='nav2_controller',
                executable='controller_server',
                name='controller_server',
                output='screen',
                respawn=use_respawn,
                respawn_delay=2.0,
                parameters=[configured_params],
                arguments=['--ros-args', '--log-level', log_level],
                remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static'), ('cmd_vel', 'cmd_vel_nav')],
            ),
            Node(
                package='nav2_lifecycle_manager',
                executable='lifecycle_manager',
                name='lifecycle_manager_navigation',
                output='screen',
                parameters=[{'autostart': autostart, 'node_names': ['controller_server']}],
                arguments=['--ros-args', '--log-level', log_level],
            ),
        ]
    )

    # Launch description
    ld = LaunchDescription()

    # Set environment variables
    ld.add_action(stdout_linebuf_envvar)

    # Declare arguments
    ld.add_action(DeclareLaunchArgument('namespace', default_value='', description='Top-level namespace'))
    ld.add_action(DeclareLaunchArgument('use_sim_time', default_value='false', description='Use simulation (Gazebo) clock if true'))
    ld.add_action(DeclareLaunchArgument('autostart', default_value='true', description='Automatically startup the nav2 stack'))
    ld.add_action(DeclareLaunchArgument('use_respawn', default_value='false', description='Whether to respawn nodes if they crash'))
    ld.add_action(DeclareLaunchArgument('log_level', default_value='info', description='Logging level'))

    # Add nodes
    ld.add_action(load_nodes)

    return ld
