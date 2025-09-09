#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import tf2_ros
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
import math
from geometry_msgs.msg import Twist
from scipy.spatial.transform import Rotation as R
import numpy as np
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data

class DWA(Node):
    def __init__(self):
        super().__init__('DWA')

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

        # Publisher cho /cmd_vel
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription = self.create_subscription(LaserScan, 'scan', self.laser_callback, qos_profile_sensor_data)
        self.human_subscriber = self.create_subscription(
            Odometry,
            '/human_global_pos',
            self.human_callback,
            10
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.timer = self.create_timer(0.001, self.timer_callback)  # Gọi mỗi 10ms
        self.vel_start = 0.3

        self.robot_radius = 1.0  # [m] for collision check

        self.set_distance_reduce = 1.5
        self.max_speed = 2.0  # [m/s]
        self.min_speed = - 0.5  # [m/s]
        self.max_yaw_rate = 90.0 * math.pi / 180.0  # [rad/s]
        self.max_accel = 0.7  # [m/ss]
        self.max_delta_yaw_rate = 90.0 * math.pi / 180.0  # [rad/ss]
        self.v_resolution = 0.1  # [m/s]
        self.yaw_rate_resolution = 0.2 * math.pi / 180.0  # [rad/s]
        self.dt = 0.1  # [s] Time tick for motion prediction
        self.predict_time = 3.0  # [s]
        self.to_goal_cost_gain = 0.15
        self.speed_cost_gain = 0.5
        self.obstacle_cost_gain = 5.0
        self.robot_stuck_flag_cons = 0.001  # constant to prevent robot stucked
        self.goal_threshold = 0.3
        # if robot_type == RobotType.rectangle
        self.robot_width = 0.5  # [m] for collision check
        self.robot_length = 1.2  # [m] for collision check
        self.robot_pos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # obstacles [x(m) y(m), ....]
        self.obs = None
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.robot_theta = 0.0

  
        
    
    def laser_callback(self, msg):
        """ Xử lý dữ liệu LaserScan mà không dùng vòng lặp """
        if self.robot_x is None:
            return
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment
        ranges = np.array(msg.ranges)  # Chuyển thành mảng NumPy để vector hóa

        # Lọc ra các giá trị hợp lệ (0.1m < range < 3m)
        valid_mask = (ranges > 0.1) & (ranges < 5.0)
        ranges = ranges[valid_mask]

        # Tạo góc tương ứng cho các điểm hợp lệ
        angles = angle_min + np.arange(len(msg.ranges)) * angle_increment
        angles = angles[valid_mask]  # Chỉ giữ góc của các điểm hợp lệ

        # Chuyển đổi sang tọa độ (x, y)
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)

        # Chuyển đổi sang hệ tọa độ odom
        x_w = self.robot_x + x * np.cos(self.robot_theta) - y * np.sin(self.robot_theta)
        y_w = self.robot_y + x * np.sin(self.robot_theta) + y * np.cos(self.robot_theta)
        
        # points = np.column_stack((x, y))  # Ghép thành mảng (N, 2)
        self.obs = np.column_stack((x_w, y_w))  # Ghép thành mảng (N, 2)

        
      
    def quaternion_to_yaw(self, quaternion):
        """Converts a quaternion to a yaw angle (in radians)."""
        return R.from_quat(quaternion).as_euler('xyz', degrees=False)[2]

    def odom_callback(self, msg):
        """Hàm callback nhận dữ liệu từ /odom"""
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        self.robot_theta = self.quaternion_to_yaw([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
        self.robot_pos = np.array([self.robot_x, self.robot_y, self.robot_theta, msg.twist.twist.linear.x,msg.twist.twist.linear.y,msg.twist.twist.angular.z])

    def human_callback(self, msg):
        """Hàm callback nhận dữ liệu từ /odom"""
        self.human_x = msg.pose.pose.position.x
        self.human_y = msg.pose.pose.position.y
        self.human_theta = self.quaternion_to_yaw([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])


        # Chuyển đổi sang hệ tọa độ odom
        self.goal_x = self.human_x + 0.0 * np.cos(self.human_theta) - 1.5 * np.sin(self.human_theta)
        self.goal_y = self.human_y + 0.0 * np.sin(self.human_theta) + 1.5 * np.cos(self.human_theta)

    def motion(self,x, u, dt):
        """
        motion model
        """

        x[2] += u[1] * dt
        x[0] += u[0] * math.cos(x[2]) * dt
        x[1] += u[0] * math.sin(x[2]) * dt

        x[3] = u[0]
        x[4] = 0.0
        x[5] = u[1]

        return x


    def calc_dynamic_window(self, x):
        """
        calculation dynamic window based on current state x
        """

        # Dynamic window from robot specification
        Vs = [self.min_speed, self.max_speed,
            -self.max_yaw_rate, self.max_yaw_rate]

        # Dynamic window from motion model
        Vd = [x[3] - self.max_accel * self.dt,
            x[3] + self.max_accel * self.dt,
            x[5] - self.max_delta_yaw_rate * self.dt,
            x[5] + self.max_delta_yaw_rate * self.dt]

        #  [v_min, v_max, yaw_rate_min, yaw_rate_max]
        dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
            max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

        return dw


    def predict_trajectory(self, x_init, v, y):
        """
        predict trajectory with an input
        """

        x = np.array(x_init)
        trajectory = np.array(x)
        time = 0
        while time <= self.predict_time:
            x = self.motion(x, [v, y], self.dt)
            trajectory = np.vstack((trajectory, x))
            time += self.dt

        return trajectory
    
    def calc_obstacle_cost(self, trajectory):
        """
        calc obstacle cost inf: collision
        """
        if self.obs is not None and len(self.obs) > 0:
            ox = self.obs[:, 0]
            oy = self.obs[:, 1]
            dx = trajectory[:, 0] - ox[:, None]
            dy = trajectory[:, 1] - oy[:, None]
            r = np.hypot(dx, dy)
            if np.array(r <= self.robot_radius).any():
                return float("Inf")

            min_r = np.min(r)
            return 1.0 / min_r  # OK
        else:
            return 0.0

    def calc_control_and_trajectory(self, x, dw, goal):
        """
        calculation final input with dynamic window
        """

        x_init = x[:]
        min_cost = float("inf")
        best_u = [0.0, 0.0]
        best_trajectory = np.array([x])

        # evaluate all trajectory with sampled input in dynamic window
        for v in np.arange(dw[0], dw[1], self.v_resolution):
            for y in np.arange(dw[2], dw[3], self.yaw_rate_resolution):

                trajectory = self.predict_trajectory(x_init, v, y)

                # calc cost
                to_goal_cost = self.to_goal_cost_gain * self.calc_to_goal_cost(trajectory, goal)
                speed_cost = self.speed_cost_gain * (self.max_speed - trajectory[-1, 3])
                ob_cost = self.obstacle_cost_gain * self.calc_obstacle_cost(trajectory)
                final_cost = to_goal_cost + speed_cost + ob_cost
            

                # search minimum trajectory
                if min_cost >= final_cost:
                    min_cost = final_cost
                    best_u = [v, y]
                    best_trajectory = trajectory
                    if abs(best_u[0]) < self.robot_stuck_flag_cons \
                            and abs(x[3]) < self.robot_stuck_flag_cons:
                        # to ensure the robot do not get stuck in
                        # best v=0 m/s (in front of an obstacle) and
                        # best omega=0 rad/s (heading to the goal with
                        # angle difference of 0)
                        best_u[1] = -self.max_delta_yaw_rate
        return best_u, best_trajectory


    def calc_to_goal_cost(self, trajectory, goal):
        """
            calc to goal cost with angle difference
        """

        dx = goal[0] - trajectory[-1, 0]
        dy = goal[1] - trajectory[-1, 1]
        error_angle = math.atan2(dy, dx)
        cost_angle = error_angle - trajectory[-1, 2]
        cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))

        return cost
    
    def dwa_control(self, x, goal):
        """
        Dynamic Window Approach control
        """
        dw = self.calc_dynamic_window(x)

        u, trajectory = self.calc_control_and_trajectory(x, dw, goal)

        return u, trajectory


    def timer_callback(self):
        """Lấy transform từ odom đến compare_point"""

        error_x = self.goal_x - self.robot_x
        error_y = self.goal_y - self.robot_y
        # error_theta = self.human_theta -self.robot_theta
        distance = math.sqrt(error_x**2+error_y**2)
        self.goal = np.array([self.goal_x, self.goal_y ])  

        if self.robot_pos[3] < self.vel_start:
            self.robot_pos[3] = self.vel_start

        u, predicted_trajectory = self.dwa_control(self.robot_pos, self.goal)
        vx = u[0]
        w = u[1]

        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = vx
        cmd_vel_msg.linear.y = 0.0
        cmd_vel_msg.angular.z = w
        self.cmd_vel_publisher.publish(cmd_vel_msg)

       



def main():
    rclpy.init()
    node = DWA()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
