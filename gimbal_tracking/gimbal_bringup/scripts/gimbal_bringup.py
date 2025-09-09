#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import serial
import numpy as np
from tf2_ros import TransformBroadcaster
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R

class SerialPublisher(Node):
    def __init__(self):
        super().__init__('gimbal_bringup')
        # Declare parameters
        self.declare_parameter('port', '/dev/ttyACM0')
        self.serial_port = self.get_parameter('port').get_parameter_value().string_value
        self.baudrate = 115200
        self.max_torque = self.declare_parameter('max_torque', 20.0).get_parameter_value().double_value
        # Declare camera position parameters (fixed x, y, z)
        self.declare_parameter('camera_x', 0.0)
        self.declare_parameter('camera_y', 0.0)
        self.declare_parameter('camera_z', 0.0)
        self.camera_x = self.get_parameter('camera_x').get_parameter_value().double_value
        self.camera_y = self.get_parameter('camera_y').get_parameter_value().double_value
        self.camera_z = self.get_parameter('camera_z').get_parameter_value().double_value
        # Initialize serial connection
        try:
            self.serial_connection = serial.Serial(self.serial_port, self.baudrate, timeout=1)
        except serial.SerialException as e:
            self.get_logger().error(f"Failed to open serial port {self.serial_port}: {e}")
            raise
        # Initialize TF broadcaster
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        self.timer = self.create_timer(0.02, self.timer_callback)
        # Initialize gimbal command and flag
        self.gimbal_cmd = None
        self.get_gimbal_cmd = False
        # Create subscription to control_gimbal topic
        self.subscription = self.create_subscription(
            Float32,
            'control_gimbal',
            self.gimbal_control_callback,
            10)
        # Create publisher for gimbal encoder data
        self.encoder_publisher = self.create_publisher(Float32, 'gimbal_encoder', 10)


    def euler_to_quaternion(self,roll, pitch, yaw):
        """Convert Euler angles to quaternion."""
        return R.from_euler('xyz', [roll, pitch, yaw]).as_quat()
    
    def timer_callback(self):
        # Send gimbal command over serial if available
        if self.get_gimbal_cmd and self.gimbal_cmd is not None:
            try:
                self.serial_connection.write(self.gimbal_cmd)
                # self.get_logger().info(f"Sent command: {self.gimbal_cmd.decode().strip()}")
                self.get_gimbal_cmd = False # Reset flag after sending
            except serial.SerialException as e:
                self.get_logger().error(f"Failed to send command: {e}")
        # Read data from serial port
        try:
            # Read a line from the serial port (assumes data ends with \n)
            line = self.serial_connection.readline().decode().strip()
            if line.startswith('p'):
                # Extract the numeric part after 'p' and convert to float (assuming angle in degrees)
                encoder_angle = float(line[1:])
                # self.get_logger().info(f"Received encoder data: {encoder_angle}")
                # Publish the received data to the gimbal_encoder topic
                msg = Float32()
                msg.data = encoder_angle
                self.encoder_publisher.publish(msg)
                # Broadcast TF from base_link to camera
                t = TransformStamped()
                t.header.stamp = self.get_clock().now().to_msg()
                t.header.frame_id = 'base_link'
                t.child_frame_id = 'camera_gimbal'
                # Set fixed position
                t.transform.translation.x = self.camera_x
                t.transform.translation.y = self.camera_y
                t.transform.translation.z = self.camera_z
                # Convert encoder angle (degrees) to quaternion (rotation around z-axis for yaw)
                # print(encoder_angle)
                quaternion = self.euler_to_quaternion(0.0, 0.0, encoder_angle)
                # print(quaternion)
                t.transform.rotation.x = quaternion[0]
                t.transform.rotation.y = quaternion[1]
                t.transform.rotation.z = quaternion[2]
                t.transform.rotation.w = quaternion[3]
                # Send the transform
                self.tf_broadcaster.sendTransform(t)
        except serial.SerialException as e:
            self.get_logger().error(f"Failed to read from serial: {e}")
    def gimbal_control_callback(self, msg):
        clamped_value = np.clip(msg.data, -self.max_torque, self.max_torque)
        # self.get_logger().info(f"Clamped value: {clamped_value}")
        self.gimbal_cmd = f"T{clamped_value}\n".encode()
        self.get_gimbal_cmd = True
    def destroy_node(self):
        # Close serial connection on node destruction
        if self.serial_connection.is_open:
            self.serial_connection.close()
            self.get_logger().info("Serial connection closed")
        super().destroy_node()
def main(args=None):
    rclpy.init(args=args)
    try:
        serial_publisher = SerialPublisher()
        rclpy.spin(serial_publisher)
    except KeyboardInterrupt:
        print("Shutting down node")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'serial_publisher' in locals():
            serial_publisher.destroy_node()
        rclpy.shutdown()
if __name__ == '__main__':
    main()