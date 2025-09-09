# Adaptive Human Group Companionship for Social Robots Using Vision-Language Models

These packages were developed by [Thanh](https://sites.google.com/view/vuthanhcdt/home) from the [Networked Robotic Systems Laboratory](https://sites.google.com/site/yenchenliuncku). If you use any packages from this repository, please cite this repository and our team.

---

## Overview

This project is built on the Agilex Scout Mini Omni platform. The directory structure is as follows:
```
ral/
├── genbot                          // Original packages
│   ├──  scout_ros2                 // Genbot-specific ROS2 packages
|   |   ├── scout_base              // Core functions for Genbot
|   |   ├── scout_msgs              // Message definitions for Genbot
|   |   ├── scout_simulation        // Simulation environment for Genbot
|   |   ├── ugv_sdk                 // Data transmission protocol for Genbot
|   |   ├── actor_control           // Control actor on Gazebo
|   ├── velodyne                    // Velodyne-related packages
|   ├── amfitrack                   // Amfitrack packages
├── vlm_strategy                    // VLM strategy
├── mppi_controller                 // Human-companion controller
├── README.md

```

## Install Dependent ROS Packages

Before proceeding, ensure you have the following installed and configured:  
- **Jetson 6** with a compatible **JetPack** and **CUDA** version (required for Jetson Orin)  
- **[ZED X Drivers](https://www.stereolabs.com/en-tw/developers/drivers)** and **[ZED X SDK](https://www.stereolabs.com/en-tw/developers/release)**  
- **[ROS2 Humble](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)**  

This project has been thoroughly tested on:  
- **Ubuntu 24.04** with **ROS2 Jazzy** (on the host computer)  
- **JetPack 6.2** (on the robot)  

For optimal compatibility, it is highly recommended to use this setup. To install the required ROS packages, run the following command:  


```bash
sudo apt-get install ros-$ROS_DISTRO-joy ros-$ROS_DISTRO-teleop-twist-joy \
  ros-$ROS_DISTRO-teleop-twist-keyboard ros-$ROS_DISTRO-laser-proc \
  ros-$ROS_DISTRO-urdf ros-$ROS_DISTRO-xacro \
  ros-$ROS_DISTRO-compressed-image-transport ros-$ROS_DISTRO-rqt\
  ros-$ROS_DISTRO-interactive-markers \
  ros-$ROS_DISTRO-slam-toolbox\
  ros-$ROS_DISTRO-rqt ros-$ROS_DISTRO-rqt-common-plugins\
  ros-$ROS_DISTRO-sophus\
  ros-$ROS_DISTRO-robot-localization\
  ros-$ROS_DISTRO-realsense2-camera\
  ros-$ROS_DISTRO-realsense2-description\
  build-essential git cmake libasio-dev\
  ros-$ROS_DISTRO-tf2-geometry-msgs\
  ros-$ROS_DISTRO-eigen-stl-containers\
  ros-$ROS_DISTRO-ament-cmake-clang-format\
  ros-$ROS_DISTRO-nmea-msgs\
  ros-$ROS_DISTRO-mavros\
  ros-$ROS_DISTRO-navigation2\
  ros-$ROS_DISTRO-nav2-bringup\
  ros-$ROS_DISTRO-bondcpp\
  ros-$ROS_DISTRO-ompl\
  ros-$ROS_DISTRO-pcl-ros\
  ros-$ROS_DISTRO-sensor-msgs-py\
  ros-$ROS_DISTRO-tf2-tools\
  ros-$ROS_DISTRO-robot-state-publisher\
  ros-$ROS_DISTRO-ros-core\
  ros-$ROS_DISTRO-geometry2\
  ros-$ROS_DISTRO-tf2-sensor-msgs\
  ros-$ROS_DISTRO-spatio-temporal-voxel-layer\
  libompl-dev\
  xterm\
  ros-$ROS_DISTRO-zed-msgs\
  libpcap-dev\
  ros-$ROS_DISTRO-plotjuggler-ros\
  ros-$ROS_DISTRO-mola \
  ros-$ROS_DISTRO-mola-state-estimation \
  ros-$ROS_DISTRO-mola-lidar-odometry\
  ros-${ROS_DISTRO}-ros-gz\
  ros-${ROS_DISTRO}-velodyne\
  ros-${ROS_DISTRO}-tf2-geometry-msgs\
```

## Install Genbot Packages

Run the following commands to set up the workspace and install the required packages:
```bash
mkdir -p ~/icra_ws/src
cd ~/icra_ws/src/
git clone git@github.com:vuthanhcdt/icra26.git
cd ~/icra_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
echo "source ~/ral_ws/install/setup.bash" >> ~/.bashrc
pip3 install pytorch-mppi
echo "export OPENAI_API_KEY="your_api_key_here"" >> ~/.bashrc
```

## Simulation 
#### Simulation S1
```bash
ros2 launch scout_simulation s1.launch.py 
ros2 launch actor_control actor_control.launch.py config:=s1_params.yaml
ros2 launch actor_control actor_control.launch.py config:=s1_ped_params.yaml
ros2 launch mppi_controller mppi_controller.launch.py config:=params_gazebo_s1.yaml
```


#### Gazebo GUI Tip
```bash
# Follow a model in the GUI (e.g., a model named "human")
gz service -s /gui/follow --reqtype gz.msgs.StringMsg  --reptype gz.msgs.Boolean  --timeout 2000   --req 'data: "human"'
# Set camera offset relative to the followed model
gz service -s /gui/follow/offset --reqtype gz.msgs.Vector3d  --reptype gz.msgs.Boolean --timeout 2000  --req 'x: -3  y: 0  z: 2'
```

## Experiment 
```bash
ros2 launch scout_simulation robot_experiment.launch.py 
ros2 launch scout_base scout_mini_omni_base.launch.py publish_tf:=false
ros2 launch gimbal_bringup gimbal_bringup.launch.py 
ros2 launch lidar3d velodyne-all-nodes-VLP16-launch.py
ros2 launch lidar3d rviz_MID360_launch.py 
ros2 launch zed_multi_camera zed_multi_camera.launch.py cam_names:='[zed_vlm, zed_gimbal]' cam_models:='[zedx,zedx]' cam_serials:='[44820006,43870948]' disable_tf:=False body_tracking:='[false,true]' object_detection:='[true,false]'
ros2 launch amfitrack amfitrack.launch.py
ros2 launch gimbal_bringup gimbal_tracking.launch.py 
ros2 launch mppi_human_following mppi_human_following.launch.py 
ros2 launch mppi_human_following mppi_human_following_local.launch.py 
python3 ral.py
ros2 run joy joy_node
```

## TODO
- [ ] 
- [ ] 
- [ ] 
