#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import tf2_ros
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
import math
from geometry_msgs.msg import Twist
from scipy.spatial.transform import Rotation as R

class PIDController:
    """Hàm PID với đầu vào là sai số và hệ số"""
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0.0
        self.integral = 0.0

    def compute(self, error, dt):
        """Tính toán đầu ra PID"""
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output


class PID(Node):
    def __init__(self):
        super().__init__('pid')

        # Subscriber cho topic /odom
        self.odom_subscriber = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        self.human_subscriber = self.create_subscription(
            Odometry,
            '/human_global_pos',
            self.human_callback,
            10
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.timer = self.create_timer(0.01, self.lookup_transform)  # Gọi mỗi 10ms
        self.pid = PIDController(kp=5.0, ki=0.0, kd=0.0)  # Khởi tạo PID

        # Publisher cho /cmd_vel
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0  # Góc quay của robot


        self.human_x = 0.0
        self.human_y = 0.0
        self.human_theta = 0.0  # Góc quay của robot

      
    def quaternion_to_yaw(self, quaternion):
        """Converts a quaternion to a yaw angle (in radians)."""
        return R.from_quat(quaternion).as_euler('xyz', degrees=False)[2]

    def odom_callback(self, msg):
        """Hàm callback nhận dữ liệu từ /odom"""
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        self.robot_theta = self.quaternion_to_yaw([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])

    def human_callback(self, msg):
        """Hàm callback nhận dữ liệu từ /odom"""
        self.human_x = msg.pose.pose.position.x
        self.human_y = msg.pose.pose.position.y
        self.human_theta = self.quaternion_to_yaw([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])

    def lookup_transform(self):
        """Lấy transform từ odom đến compare_point"""
        try:
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform('odom', 'compare_point', now)
            point_x = trans.transform.translation.x
            point_y = trans.transform.translation.y

            error_x = point_x - self.robot_x
            error_y = point_y - self.robot_y
            error_theta = self.human_theta -self.robot_theta

            vx = self.pid.compute(error_x, 0.01)  # dt = 10ms
            vy = self.pid.compute(error_y, 0.01)  # dt = 10ms
            w =  self.pid.compute(error_theta, 0.01)  # dt = 10ms

            max_vx = 1.0
            max_vy = 0.5
            max_w = 1.5

            vx = max(-max_vx, min(vx, max_vx))
            vy = max(-max_vy, min(vy, max_vy))
            w = max(-max_w, min(w, max_w))



            cmd_vel_msg = Twist()
            cmd_vel_msg.linear.x = vx
            cmd_vel_msg.linear.y = vy
            cmd_vel_msg.angular.z = w
            self.cmd_vel_publisher.publish(cmd_vel_msg)


            # print(vx,vy,w)


        except tf2_ros.LookupException:
            self.get_logger().warn("TF not available yet")
        except tf2_ros.ConnectivityException:
            self.get_logger().warn("TF connectivity issue")
        except tf2_ros.ExtrapolationException:
            self.get_logger().warn("TF extrapolation issue")


def main():
    
    rclpy.init()
    node = PID()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
