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
from scipy.optimize import minimize
from nav_msgs.msg import OccupancyGrid

class Robot(object):
    def __init__(
        self, x: float, y: float
    ):
        self.p: np.ndarray = np.reshape([x, y, 0], (3, 1))
        self.v: np.ndarray = np.reshape([0, 0, 0], (3, 1))


    def update(self, v: float, w: float, dt: float = 0.5):
        self.v = np.reshape([v, 0, w], (3, 1))
        mat: np.ndarray = np.array([
            [np.cos(self.p[2, 0]) * dt, 0],
            [np.sin(self.p[2, 0]) * dt, 0],
            [0, dt]
        ])
        vel: np.ndarray = self.v[[0, 2], :]
        self.p += mat @ vel
    
    def get_state(self):
        return self.p.copy(), self.v.copy()
    
    def set_state(self, p: np.ndarray, v: np.ndarray) :
        self.p, self.v = p, v
    

class MPCController(object):
    def __init__(self, window: int = 1) -> None:
        self.window: int = window
        self.Q: np.ndarray = np.diag([1.0, 1.0])    # state cost matrix
        self.R: np.ndarray = np.diag([0.01, 0.01])  # input cost matrix
        self.Rd: np.ndarray = np.diag([0.01, 1.0])  # input difference cost matrix

    def update(self, target, robot, obs ):
        bounds: list = [(-1.0, 1.0), (-np.pi, np.pi)] * self.window
        ctrl = minimize(
            self._cost,
            x0=np.zeros((self.window * 2)),
            args=(np.array(target), robot, obs),
            method="SLSQP",
            bounds=bounds,
        )
        return ctrl.x[0], ctrl.x[1]
    
              

    def _cost(self, u_k, target, robot, obs ):
        state = robot.get_state() # lay vi tri
        u_k = u_k.reshape(self.window, 2).T
        x_k = np.zeros((2, self.window + 1))

        # input cost
        cost = np.sum(self.R @ (u_k**2))

        # state cost
        for i in range(self.window):
            robot.update(u_k[0, i], u_k[1, i])
            p, _ = robot.get_state()
            x_k[:, i] = p[:2].flatten()
        # print(x_k[:, :-1])
        # print(x_k[:, :-1][0])

        # Thêm None vào để phát triển x_k thành (2, 5, 1), sau đó phép trừ sẽ có kích thước (2, 5, 100)
        if obs is not None: 
            x_k_reshaped = x_k[:, :-1].T
            dists = np.linalg.norm(x_k_reshaped[:, None, :] - obs[None, :, :], axis=2)  # (5, 100)
            if dists.size > 0:
                min_dists = np.min(dists, axis=1)  # (5,)
                _cost_obs = np.sum(min_dists)
                cost_obs=10000.0*np.exp(-_cost_obs)
                cost += cost_obs
                # print(1000000.0*np.exp(-cost_obs))
        
        state_error = target[:, np.newaxis] - x_k[:, :-1]
        cost += np.sum(self.Q @ (state_error**2))
        # print(cost)

        # input difference cost
        input_diff: np.ndarray = np.diff(u_k, axis=1)
        cost += np.sum(self.Rd @ (input_diff**2))

        robot.set_state(state[0], state[1])

        return float(cost)


class MPC(Node):
    def __init__(self):
        super().__init__('MPC')

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

        self.create_subscription(OccupancyGrid, '/local_costmap/costmap', self.map_callback, 2)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.timer = self.create_timer(0.001, self.timer_callback)  # Gọi mỗi 10ms
        self.dt = 0.1  # [s] Time tick for motion prediction
        self.predict_time = 1.0  # [s]
        self.K = int(self.predict_time / self.dt)

        # obstacles [x(m) y(m), ....]
        # self.obs = None
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0

        # self.goal_x = 0.0
        # self.goal_y = 0.0

        self.goal_x = -7.0
        self.goal_y = -1.0

        self.v_init = 0.0
        self.w_init = 0.0
        self.u_init =  np.zeros((self.K * 2))

        self.max_v = 1.0  # [m/s]
        self.min_v = - 1.0  # [m/s]
        self.max_w = 90.0 * math.pi / 180.0  # [rad/s]
        self.dt = 0.5  # [s] Time tick for motion prediction
        self.weight_x = 0.01
        self.weight_y = 0.5
        self.weight_v = 0.0
        self.weight_w = 0.0

        self.mpc_ctrl = MPCController(5)
        self.mpc_robot = Robot(-7.0, 3.5)

        self.robot_pos: np.ndarray = np.reshape([0.0, 0.0, 0.0], (3, 1))
        self.robot_vel: np.ndarray = np.reshape([0.0, 0.0, 0.0], (3, 1))

        self.obs = None
        self.map_origin_x = 0.0
        self.map_origin_y = 0.0




  

    def laser_callback(self, msg):
        """ Xử lý dữ liệu LaserScan mà không dùng vòng lặp """
        if self.robot_x is None:
            return
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment
        # print(angle_increment)
        
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
        self.robot_pos = np.reshape([self.robot_x, self.robot_y, self.robot_theta], (3, 1))
        self.robot_vel: np.ndarray = np.reshape([msg.twist.twist.linear.x,msg.twist.twist.linear.y,msg.twist.twist.angular.z], (3, 1))

    def map_callback(self, msg):
        """Callback function to process the occupancy grid map when it is received."""
        self.map_resolution =  msg.info.resolution
        self.map_origin_x =  msg.info.origin.position.x
        self.map_origin_y =  msg.info.origin.position.y
        self.map_height =  msg.info.height
        self.map_width =  msg.info.width
        self.obs = msg.data

        # print( self.map_resolution,self.map_origin_x,self.map_origin_y ,self.map_height,self.map_width)


    def human_callback(self, msg):
        """Hàm callback nhận dữ liệu từ /odom"""
        self.human_x = msg.pose.pose.position.x
        self.human_y = msg.pose.pose.position.y
        self.human_theta = self.quaternion_to_yaw([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])


        # Chuyển đổi sang hệ tọa độ odom
        self.goal_x = self.human_x + 0.0 * np.cos(self.human_theta) - 1.5 * np.sin(self.human_theta)
        self.goal_y = self.human_y + 0.0 * np.sin(self.human_theta) + 1.5 * np.cos(self.human_theta)



    def timer_callback(self):
        """Lấy transform từ odom đến compare_point"""

        # error_x = self.goal_x - self.robot_x
        # error_y = self.goal_y - self.robot_y
        # # error_theta = self.human_theta -self.robot_theta
        # distance = math.sqrt(error_x**2+error_y**2)
        # self.goal = np.array([self.goal_x, self.goal_y ])  

        # if self.robot_pos[3] < self.vel_start:
        #     self.robot_pos[3] = self.vel_start

        
        self.mpc_robot.set_state(self.robot_pos, self.robot_vel)
        target = (self.goal_x,self.goal_y)
        # target = (0.0, 0.0)
        vx, w = self.mpc_ctrl.update(target, self.mpc_robot, self.obs)
        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = vx
        cmd_vel_msg.linear.y = 0.0
        cmd_vel_msg.angular.z = w
        self.cmd_vel_publisher.publish(cmd_vel_msg)

       



def main():
    rclpy.init()
    node = MPC()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
