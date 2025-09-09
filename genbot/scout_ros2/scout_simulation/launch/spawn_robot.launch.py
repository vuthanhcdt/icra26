import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import AppendEnvironmentVariable

def generate_launch_description():

    pkg_urdf_path = get_package_share_directory('scout_simulation')

    rviz_launch_arg = DeclareLaunchArgument(
        'rviz', default_value='true',
        description='Open RViz.'
    )

    world_arg = DeclareLaunchArgument(
        'world', default_value='empty.sdf',
        description='Name of the Gazebo world file to load'
    )

    model_arg = DeclareLaunchArgument(
        'model', default_value='scout_mini.urdf.xacro',
        description='Name of the URDF description to load'
    )

    # Define the path to your URDF or Xacro file
    urdf_file_path = PathJoinSubstitution([
        pkg_urdf_path,  # Replace with your package name
        "urdf","robots",
        LaunchConfiguration('model')  # Replace with your URDF or Xacro file
    ])

    world_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_urdf_path, 'launch', 'world.launch.py'),
        ),
        launch_arguments={
        'world': LaunchConfiguration('world'),
        }.items()
    )

    # Launch rviz
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', os.path.join(pkg_urdf_path, 'rviz', 'scout_simulation.rviz')],
        condition=IfCondition(LaunchConfiguration('rviz')),
        parameters=[
            {'use_sim_time': True},
        ]
    )

    # Spawn the URDF model using the `/world/<world_name>/create` service
    spawn_urdf_node = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-name", "scout_mini",
            "-topic", "robot_description",
            "-x", "0.0", "-y", "0.0", "-z", "0.07", "-Y", "0.0"  # Initial spawn position
        ],
        output="screen",
        parameters=[
            {'use_sim_time': True},
        ]
    )

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[
            {'robot_description': Command(['xacro', ' ', urdf_file_path]),
             'use_sim_time': True},
        ],
        remappings=[
            ('/tf', 'tf'),
            ('/tf_static', 'tf_static')
        ]
    )

    # Node to bridge messages like /cmd_vel and /odom
    gz_bridge_node = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist",
            "/cmd_vel_actor@geometry_msgs/msg/Twist@gz.msgs.Twist",
            "/mode_position_actor@std_msgs/msg/Bool@gz.msgs.Boolean",
            "/pose_control_actor@geometry_msgs/msg/Pose@gz.msgs.Pose",
            "/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry",
            "/joint_states@sensor_msgs/msg/JointState@gz.msgs.Model",
            "/tf@tf2_msgs/msg/TFMessage@gz.msgs.Pose_V",
            # "/camera_follow/image@sensor_msgs/msg/Image@gz.msgs.Image",
            # "/camera_follow/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo",
            "/imu@sensor_msgs/msg/Imu@gz.msgs.IMU", 
            "/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan",             
            "/scan/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked", 
            "/realsense/image@sensor_msgs/msg/Image@gz.msgs.Image",
            "/realsense/depth_image@sensor_msgs/msg/Image@gz.msgs.Image",
            "/realsense/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked",   
            "/human_local_pose@geometry_msgs/msg/Pose@gz.msgs.Pose", 
            "/human_global_pose@geometry_msgs/msg/Pose@gz.msgs.Pose", 
        ],
        output="screen",
        parameters=[
            {'use_sim_time': True},
        ]
    )

    # joint_state_publisher_gui_node = Node(
    #     package='joint_state_publisher_gui',
    #     executable='joint_state_publisher_gui',
    # )


    # Add a gazebo library path here
    set_env_vars_resources = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        os.path.join(
            get_package_share_directory('scout_simulation'),
            'models'))
    
    set_plugin_vars_resources = AppendEnvironmentVariable(
        'GZ_SIM_SYSTEM_PLUGIN_PATH',
        os.path.join(
            get_package_share_directory('scout_simulation'),
            'plugin/build'))


    launchDescriptionObject = LaunchDescription()
    launchDescriptionObject.add_action(set_plugin_vars_resources)
    launchDescriptionObject.add_action(set_env_vars_resources)
    launchDescriptionObject.add_action(rviz_launch_arg)
    launchDescriptionObject.add_action(world_arg)
    launchDescriptionObject.add_action(model_arg)
    launchDescriptionObject.add_action(world_launch)
    launchDescriptionObject.add_action(rviz_node)
    launchDescriptionObject.add_action(spawn_urdf_node)
    launchDescriptionObject.add_action(robot_state_publisher_node)
    launchDescriptionObject.add_action(gz_bridge_node)

    return launchDescriptionObject
