#!/usr/bin/env python3

import rclpy
import math
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import Pose2D, TransformStamped
from tf2_ros import StaticTransformBroadcaster
from scipy.spatial.transform import Rotation as R
from nav_msgs.msg import Odometry
import amfiprot
import amfiprot_amfitrack as amfitrack
import time
from nav_msgs.msg import Odometry


class Gen3Publisher(Node):
    def __init__(self):
        super().__init__('amfitrack')

        # Initialize QoS profile and parameters
        self.qos = QoSProfile(depth=1)
        self.VENDOR_ID = 0xC17
        self.PRODUCT_ID_SENSOR = 0xD12
        self.PRODUCT_ID_SOURCE = 0xD01

        self.human_global = Odometry()  # Robot-to-human relative position and angle
        self.relative_robot_human = Pose2D()
        self.relativeState = Pose2D() 

        self.broadcaster = StaticTransformBroadcaster(self)
        self.transformStamped = TransformStamped()

        # Declare ROS parameters
        self.relative_topic = self.declare_parameter('relativePos', 'human_robot_pos').get_parameter_value().string_value
        self.tf_frame_id = self.declare_parameter('tf_frame_id', 'odom').get_parameter_value().string_value
        self.base_link = self.declare_parameter('base_link', 'base_link').get_parameter_value().string_value

        self.tf_child_name = self.declare_parameter('tf_child_name', 'human_link').get_parameter_value().string_value
        self.delta_t = self.declare_parameter('delta_t', 0.05).get_parameter_value().double_value
        self.num_step = self.declare_parameter('num_step', 30).get_parameter_value().integer_value
        self.robot_sub = self.create_subscription(Odometry,'/odom', self.odom_callback,5)

        # Set up publishers
        self.relative_pub = self.create_publisher(Pose2D, self.relative_topic, self.qos)
        self.robot_pub = self.create_publisher(Pose2D, 'robot_human_pos', self.qos)
        self.human_global_pub = self.create_publisher(Odometry, 'human_global_pos', self.qos)

        # Initialize Amfitrack device
        self.device = None
        self.setup_successful = self.setup_gen3()

        if self.setup_successful:
            self.timer = self.create_timer(0.001, self.timer_callback)  # Timer for periodic execution
        else:
            self.get_logger().error("Failed to set up Amfitrack device.")

        # Initialize variables to track previous position and time
        self.prev_time = time.time()
        self.prev_position = (0.0,0.0,0.0)
        self.robot_theta=0.0
        self.robot_x=0.0
        self.robot_y=0.0

        self.linear_x = 0.0
        self.linear_y = 0.0
        self.angular_velocity = 0.0

        self.human_global.header.frame_id=self.tf_frame_id
        self.human_global.child_frame_id=self.tf_child_name

        self.velocity_x_history = []  
        self.velocity_y_history = []  


    def odom_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        self.robot_theta = self.quaternion_to_yaw([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])


    def setup_gen3(self):
        """Set up the Amfitrack device."""
        conn = None
        try:
            conn = amfiprot.USBConnection(self.VENDOR_ID, self.PRODUCT_ID_SENSOR)
        except Exception:
            try:
                conn = amfiprot.USBConnection(self.VENDOR_ID, self.PRODUCT_ID_SOURCE)
            except Exception:
                self.get_logger().error("No Amfitrack device found.")
                return False

        # Discover nodes
        nodes = conn.find_nodes()
        self.get_logger().info(f"Found {len(nodes)} node(s).")
        for node in nodes:
            self.get_logger().info(f"[{node.tx_id}] {node.name}")

        if nodes:
            self.device = amfitrack.Device(nodes[0 if len(nodes) == 1 else 1])
            conn.start()
            # self.device.calibrate()
            return True
        else:
            self.get_logger().error("No nodes found on the Amfitrack device.")
            return False

    @staticmethod
    def euler_to_quaternion(roll, pitch, yaw):
        """Convert Euler angles to quaternion."""
        return R.from_euler('xyz', [roll, pitch, yaw]).as_quat()

    @staticmethod
    def quaternion_to_yaw(quaternion):
        """Extract yaw angle from a quaternion."""
        return R.from_quat(quaternion).as_euler('xyz')[2]
    
    @staticmethod
    def yaw_to_quaternion(yaw: float) -> np.ndarray:
        """Converts a yaw angle (in radians) to a quaternion."""
        return R.from_euler('xyz', [0.0, 0.0, yaw], degrees=False).as_quat()

    def timer_callback(self):
        """Periodic callback for processing Amfitrack data."""
        if not self.device or not self.device.packet_available():
            return

        packet = self.device.get_packet()
        # print(packet)
        if isinstance(packet.payload, amfitrack.payload.EmfImuFrameIdPayload):
            payload = packet.payload
            self.update_transforms(payload)
            self.publish_relative_position(payload)
        else:
            self.get_logger().info("Please restart the node and turn on the tag first")

    def update_transforms(self, payload):
        """Update TF transformations."""
        self.transformStamped.header.stamp = self.get_clock().now().to_msg()
        self.transformStamped.header.frame_id = self.base_link
        self.transformStamped.child_frame_id = self.tf_child_name

        self.transformStamped.transform.translation.x = payload.emf.pos_x * 0.001
        self.transformStamped.transform.translation.y = payload.emf.pos_y * 0.001
        self.transformStamped.transform.translation.z = 0.0
        self.transformStamped.transform.rotation.z = payload.emf.quat_z
        self.transformStamped.transform.rotation.w = payload.emf.quat_w

        self.broadcaster.sendTransform(self.transformStamped)

    def publish_relative_position(self, payload):
        """Calculate and publish the robot-to-human relative position."""
        x_human = payload.emf.pos_x * 0.001
        y_human = payload.emf.pos_y * 0.001
        theta_human = self.quaternion_to_yaw([0.0, 0.0, payload.emf.quat_z, payload.emf.quat_w])
        x_human_trimmed = float(f"{x_human:.2f}") 
        y_human_trimmed = float(f"{y_human:.2f}")
        theta_human_trimmed = float(f"{theta_human:.2f}")
        
        current_time = time.time()
        delta_time = current_time - self.prev_time
        if delta_time > self.delta_t:
            # Calculate velocity in x, y
            delta_x = x_human_trimmed - self.prev_position[0]
            delta_y = y_human_trimmed - self.prev_position[1]
            
            # Linear velocity
            dot_x = delta_x / delta_time
            dot_y = delta_y / delta_time
            # Calculate angular velocity (change in theta)
            delta_theta = theta_human_trimmed - self.prev_position[2]
            self.angular_velocity = delta_theta / delta_time
            linear_x = dot_x * math.cos(theta_human_trimmed) + dot_y * math.sin(theta_human_trimmed)
            linear_y = -dot_x * math.sin(theta_human_trimmed) + dot_y * math.cos(theta_human_trimmed)

            # Add current linear velocities to history
            self.velocity_x_history.append(linear_x)
            self.velocity_y_history.append(linear_y)

            # Keep only the last 10 velocities for smoothing
            if len(self.velocity_x_history) > self.num_step:
                self.velocity_x_history.pop(0)
            if len(self.velocity_y_history) > self.num_step:
                self.velocity_y_history.pop(0)

            #Compute the average of the last 10 velocities
            self.linear_x = sum(self.velocity_x_history) / len(self.velocity_x_history)
            self.linear_y = sum(self.velocity_y_history) / len(self.velocity_y_history)

            self.prev_time = current_time
            self.prev_position = [x_human_trimmed, y_human_trimmed, theta_human_trimmed]
            # print(dot_x,delta_time,self.linear_x)

        self.relativeState.x = x_human
        self.relativeState.y = y_human
        self.relativeState.theta = theta_human

        self.relative_robot_human.x = - x_human * math.cos(theta_human) - y_human * math.sin(theta_human)
        self.relative_robot_human.y = x_human * math.sin(theta_human) -  y_human * math.cos(theta_human)
        self.relative_robot_human.theta = -theta_human

        self.relative_pub.publish(self.relativeState)
        self.robot_pub.publish(self.relative_robot_human)

        human_global_x = self.robot_x + (x_human* math.cos(self.robot_theta) - y_human * math.sin(self.robot_theta))
        human_global_y = self.robot_y + (x_human * math.sin(self.robot_theta) + y_human * math.cos(self.robot_theta))
        human_global_theta = theta_human + self.robot_theta

        self.human_global.pose.pose.position.x = human_global_x
        self.human_global.pose.pose.position.y = human_global_y

        self.human_global.header.stamp = self.get_clock().now().to_msg()

        quat_human = self.yaw_to_quaternion(human_global_theta)

        self.human_global.twist.twist.linear.x =  self.linear_x 
        self.human_global.twist.twist.linear.y =  self.linear_y
        self.human_global.twist.twist.angular.z = self.angular_velocity

        self.human_global.pose.pose.orientation.x = quat_human[0]
        self.human_global.pose.pose.orientation.y = quat_human[1]
        self.human_global.pose.pose.orientation.z = quat_human[2]
        self.human_global.pose.pose.orientation.w = quat_human[3]

        self.human_global_pub.publish(self.human_global)

def main(args=None):
    rclpy.init(args=args)
    node = Gen3Publisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
