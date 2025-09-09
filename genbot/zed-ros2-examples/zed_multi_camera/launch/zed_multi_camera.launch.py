# Copyright 2024 Stereolabs
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from ament_index_python.packages import get_package_share_directory
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    OpaqueFunction,
    IncludeLaunchDescription,
    LogInfo
)
from launch.substitutions import (
    LaunchConfiguration,
    Command,
    TextSubstitution
)
from launch_ros.actions import (
    Node,
    ComposableNodeContainer
)

def parse_array_param(param):
    str = param.replace('[', '')
    str = str.replace(']', '')
    str = str.replace(' ', '')
    arr = str.split(',')

    return arr

def launch_setup(context, *args, **kwargs):

    # List of actions to be launched
    actions = []

    namespace_val = 'zed_multi'
    
    # URDF/xacro file to be loaded by the Robot State Publisher node
    multi_zed_xacro_path = os.path.join(
    get_package_share_directory('zed_multi_camera'),
    'urdf',
    'zed_multi.urdf.xacro')

    names = LaunchConfiguration('cam_names')
    models = LaunchConfiguration('cam_models')
    serials = LaunchConfiguration('cam_serials')
    ids = LaunchConfiguration('cam_ids')
    body_trackings = LaunchConfiguration('body_tracking')
    object_detections = LaunchConfiguration('object_detection')

    disable_tf = LaunchConfiguration('disable_tf')

    names_arr = parse_array_param(names.perform(context))
    models_arr = parse_array_param(models.perform(context))
    serials_arr = parse_array_param(serials.perform(context))
    ids_arr = parse_array_param(ids.perform(context))
    body_tracking_arr = parse_array_param(body_trackings.perform(context))
    object_detection_arr = parse_array_param(object_detections.perform(context))
    disable_tf_val = disable_tf.perform(context)

    num_cams = len(names_arr)

    if (num_cams != len(models_arr)):
        return [
            LogInfo(msg=TextSubstitution(
                text='The `cam_models` array argument must match the size of the `cam_names` array argument.'))
        ]

    if ((num_cams != len(serials_arr)) and (num_cams != len(ids_arr))):
        return [
            LogInfo(msg=TextSubstitution(
                text='The `cam_serials` or `cam_ids` array argument must match the size of the `cam_names` array argument.'))
        ]

    if (num_cams != len(body_tracking_arr)):
        return [
            LogInfo(msg=TextSubstitution(
                text='The `body_tracking_arr` array argument must match the size of the `cam_names` array argument.'))
        ]

    if (num_cams != len(object_detection_arr)):
        return [
            LogInfo(msg=TextSubstitution(
                text='The `object_detection_arr` array argument must match the size of the `cam_names` array argument.'))
        ]
    
    # ROS 2 Component Container
    container_name = 'zed_multi_container'
    distro = os.environ['ROS_DISTRO']
    if distro == 'foxy':
        # Foxy does not support the isolated mode
        container_exec='component_container'
    else:
        container_exec='component_container_isolated'
    
    info = '* Starting Composable node container: /' + namespace_val + '/' + container_name
    actions.append(LogInfo(msg=TextSubstitution(text=info)))

    zed_container = ComposableNodeContainer(
        name=container_name,
        namespace=namespace_val,
        package='rclcpp_components',
        executable=container_exec,
        arguments=['--ros-args', '--log-level', 'info'],
        output='screen',
    )
    actions.append(zed_container)

    # Set the first camera idx
    cam_idx = 0

    for name in names_arr:
        model = models_arr[cam_idx]
        if len(serials_arr) == num_cams:
            serial = serials_arr[cam_idx]
        else:
            serial = '0'

        if len(ids_arr) == num_cams:
            id = ids_arr[cam_idx]
        else:
            id = '-1'
        
        pose = '['

        info = '* Starting a ZED ROS2 node for camera ' + name + \
            ' (' + model        
        if(serial != '0'):
            info += ', serial: ' + serial
        elif( id!= '-1'):
            info += ', id: ' + id
        info += ')'

        actions.append(LogInfo(msg=TextSubstitution(text=info)))

        # Only the first camera send odom and map TF
        publish_tf = 'false'
        if (cam_idx == 0):
            if (disable_tf_val == 'False' or disable_tf_val == 'false'):
                publish_tf = 'true'

        # A different node name is required by the Diagnostic Updated
        node_name = 'zed_node_' + str(cam_idx)

        body_tracking = body_tracking_arr[cam_idx]
        object_detection = object_detection_arr[cam_idx]

        # Add the node
        # ZED Wrapper launch file
        zed_wrapper_launch = IncludeLaunchDescription(
            launch_description_source=PythonLaunchDescriptionSource([
                get_package_share_directory('zed_wrapper'),
                '/launch/zed_camera.launch.py'
            ]),
            launch_arguments={
                'container_name': container_name,
                'camera_name': name,
                'camera_model': model,
                'serial_number': serial,
                'camera_id': id,
                'publish_tf': publish_tf,
                'body_tracking': body_tracking,
                'object_detection': object_detection,
                # 'publish_map_tf': publish_map_tf,
                'namespace': namespace_val
            }.items()
        )
        actions.append(zed_wrapper_launch)

        cam_idx += 1

    # Create the Xacro command with correct camera names
    xacro_command = []
    xacro_command.append('xacro')
    xacro_command.append(' ')
    xacro_command.append(multi_zed_xacro_path)
    xacro_command.append(' ')
    cam_idx = 0
    for name in names_arr:
        xacro_command.append('camera_name_'+str(cam_idx)+':=')
        xacro_command.append(name)
        xacro_command.append(' ')
        cam_idx+=1

    # Robot State Publisher node
    # this will publish the static reference link for a multi-camera configuration
    # and all the joints. See 'urdf/zed_dual.urdf.xacro' as an example    
    rsp_name = 'state_publisher'
    info = '* Starting robot_state_publisher node to link all the frames: ' + rsp_name
    actions.append(LogInfo(msg=TextSubstitution(text=info)))
    multi_rsp_node = Node(
        package='robot_state_publisher',
        namespace=namespace_val,
        executable='robot_state_publisher',
        name=rsp_name,
        output='screen',
        parameters=[{
            'robot_description': Command(xacro_command).perform(context)
        }]
    )

    static_tf2_base_gimbal= Node(package="tf2_ros",
        executable="static_transform_publisher",
        output="screen" ,
        arguments=["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "camera_gimbal", "zed_gimbal_camera_link"]
    )

    static_tf2_base_amfitrack= Node(package="tf2_ros",
        executable="static_transform_publisher",
        output="screen" ,
        arguments=["0.07", "0.0", "0.3", "0.0", "0.0", "0.0", "base_link", "amfitrack_link"]
    )

    # static_tf2_base_tracking= Node(package="tf2_ros",
    #     executable="static_transform_publisher",
    #     output="screen" ,
    #     arguments=["0.3", "0.9", "0.0", "0.0", "0.0", "0.0", "human_link", "tracking"]
    # )


    # Add the robot_state_publisher node to the list of nodes to be started
    actions.append(multi_rsp_node)
    actions.append(static_tf2_base_gimbal)
    actions.append(static_tf2_base_amfitrack)
    # actions.append(static_tf2_base_tracking)

    return actions


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'cam_names',
                description='An array containing the name of the cameras, e.g. [zed_front,zed_back]'),
            DeclareLaunchArgument(
                'cam_models',
                description='An array containing the model of the cameras, e.g. [zed2i,zed2]'),
            DeclareLaunchArgument(
                'cam_serials',
                default_value=[],
                description='An array containing the serial number of the cameras, e.g. [35199186,23154724]'),
            DeclareLaunchArgument(
                'cam_ids',
                default_value=[],
                description='An array containing the ID number of the cameras, e.g. [0,1]'),
            DeclareLaunchArgument(
                'body_tracking',
                default_value=[],
                description='An array of booleans indicating whether body tracking is enabled for each body, e.g., [true, false]'),
            DeclareLaunchArgument(
                'object_detection',
                default_value=[],
                description='An array of booleans indicating whether object detection is enabled, e.g., [true, false]'),
            DeclareLaunchArgument(
                'disable_tf',
                default_value='False',
                description='If `True` disable TF broadcasting for all the cameras in order to fuse visual odometry information externally.'),
            OpaqueFunction(function=launch_setup)
        ]
    )
