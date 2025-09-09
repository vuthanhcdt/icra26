#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, Point
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R
import torch
import numpy as np
from pytorch_mppi import mppi
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path
from rclpy.time import Time

class MPPIController(Node):
    def __init__(self):
        super().__init__('mppi_controller')

        # === Params & Constants ===
        self._dtype = torch.float32
        self._device = 'cuda'
        self._goal_pos = torch.tensor([10.0, -3.0, 3.14], device=self._device, dtype=self._dtype)
        self._dt = 0.1  # time step
        self.current_state = torch.zeros(6, dtype=self._dtype, device=self._device)

        # === MPPI Controller ===
        # self.mode = "non_holonomic"  # or "holonomic"
        self.mode = "holonomic"  # or "holonomic"
        self.setup_mppi(self.mode)

        # === ROS 2 Interfaces ===
        self.setup_ros()
        self._last_time = self.get_clock().now()


    def setup_ros(self):
        self.declare_parameter('topic_velocity', 'cmd_vel')
        topic_vel = self.get_parameter('topic_velocity').value

        self.pub_vel = self.create_publisher(Twist, topic_vel, 2)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        self.local_path_publisher = self.create_publisher(Path, '/local_path', 1)
        self.sampled_path_publisher = self.create_publisher(MarkerArray, '/sampled_path', 1)

        self.twist = Twist()
        self.timer = self.create_timer(0.001, self.timer_callback)


    # ========================
    # === MPPI Setup Logic ===
    # ========================
    def setup_mppi(self, mode):
        nx = 6 #num state [x y theta vx vy w]
        horizon = 15
        num_samples = 1000
        lambda_ = 1.0

      

    
        if mode == "non_holonomic":
            self.terminal_scale = 100.0
            self.distance_weight = 100.0
            self.angle_weight = 10.0
            self.acc_min = torch.tensor([-2.0, -2.0], device=self._device, dtype=self._dtype) # accx, accw
            self.acc_max = torch.tensor([2.0,  2.0], device=self._device, dtype=self._dtype) # accx, accw
            self.v_min = torch.tensor([-2.0, -2.0], device=self._device, dtype=self._dtype) # vx, w
            self.v_max = torch.tensor([2.0,  2.0], device=self._device, dtype=self._dtype) # vx, w

            noise_sigma = torch.diag(torch.tensor([2.0, 2.0], device=self._device, dtype=torch.double)) # accx, accw
            dynamics = self.non_holonomic_dynamics
            running_cost = self.nonholonomic_cost_function
            terminal_state_cost = self.nonholonomic_terminal_cost

        elif mode == "holonomic":
            self.terminal_scale_x = 100.0
            self.terminal_scale_y = 200.0
            self.distance_weight_x = 100.0
            self.distance_weight_y = 200.0
            self.angle_weight = 200.0
            # Assuming holonomic has different controls: [vx, vy] linear velocities
            self.acc_min = torch.tensor([-2.0, -2.0, -2.0], device=self._device, dtype=self._dtype) # accx, accy, accw
            self.acc_max = torch.tensor([2.0, 2.0, 2.0], device=self._device, dtype=self._dtype) # accx, accy, accw
            self.v_min = torch.tensor([-2.0, -2.0, -2.0], device=self._device, dtype=self._dtype) # vx, vy, w
            self.v_max = torch.tensor([2.0, 2.0, 2.0], device=self._device, dtype=self._dtype) # vx, vy, w

            noise_sigma = torch.diag(torch.tensor([1.0, 1.0, 1.0], device=self._device, dtype=torch.double)) # accx, accy, accw
            dynamics = self.holonomic_dynamics
            running_cost = self.holonomic_cost_function
            terminal_state_cost = self.holonomic_terminal_cost

        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.mppi_controller = mppi.MPPI(
                dynamics=dynamics,
                running_cost=running_cost,
                terminal_state_cost = terminal_state_cost,
                nx=nx,
                noise_sigma=noise_sigma,
                num_samples=num_samples,
                horizon=horizon,
                lambda_=lambda_,
                u_min=self.acc_min,
                u_max=self.acc_max,
                device=self._device,
            )

    # ===================
    # === ROS Callback ===
    # ===================
    def odom_callback(self, msg: Odometry):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        twist = msg.twist.twist
        yaw = self.quaternion_to_yaw([ori.x, ori.y, ori.z, ori.w])
        self.current_state = torch.tensor([pos.x, pos.y, yaw, twist.linear.x, twist.linear.y, twist.angular.z], dtype=self._dtype, device=self._device)


    def publish_path(self, rollout, publisher, base):
        """Publish a single rollout path as nav_msgs/Path."""
        num_rollouts, T, nx = rollout.shape
        path_msg = Path()
        path_msg.header.frame_id = base
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for t in range(T):
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = base
            pose_stamped.header.stamp = self.get_clock().now().to_msg()
            pose_stamped.pose.position.x = float(rollout[0, t, 0].cpu())
            pose_stamped.pose.position.y = float(rollout[0, t, 1].cpu())
            pose_stamped.pose.position.z = 0.0
            pose_stamped.pose.orientation.w = 1.0
            path_msg.poses.append(pose_stamped)

        publisher.publish(path_msg)

    
    def sampled_path(self, state, base):
        """Publish the sampled path (markers for visualization) to a given publisher."""
        marker_array = MarkerArray()        
        id_path = 0
        num_rollouts, T, nx = state.shape
        # print(f"Number of rollouts: {num_rollouts}, Time steps per rollout: {T}, State dimension: {nx}")
        for rollout_idx in range(num_rollouts):
            marker = Marker()
            marker.header.frame_id = base
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'robot_paths'
            marker.id = rollout_idx
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.01
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.lifetime = rclpy.duration.Duration(seconds=1.0).to_msg()
            for t in range(T):
                pt = Point()
                pt.x = float(state[rollout_idx, t, 0].cpu())
                pt.y = float(state[rollout_idx, t, 1].cpu())
                pt.z = 0.0
                marker.points.append(pt)

            marker_array.markers.append(marker)

        self.sampled_path_publisher.publish(marker_array)

    def timer_callback(self):
        action = self.mppi_controller.command(self.current_state)
        now = self.get_clock().now()
        dt = (now - self._last_time).nanoseconds * 1e-9  # dt in seconds
        if self.mode == "non_holonomic":
            self.twist.linear.x += float(action[0]) * dt
            self.twist.angular.z += float(action[1]) * dt
            # Giới hạn (clamp)
            self.twist.linear.x = float(max(self.v_min[0], min(self.twist.linear.x, self.v_max[0])))
            self.twist.angular.z = float(max(self.v_min[1], min(self.twist.angular.z, self.v_max[1])))


        elif self.mode == "holonomic":
            self.twist.linear.x += float(action[0]) * dt
            self.twist.linear.y += float(action[1]) * dt
            self.twist.angular.z += float(action[2]) * dt

            self.twist.linear.x = float(max(self.v_min[0], min(self.twist.linear.x, self.v_max[0])))
            self.twist.linear.y = float(max(self.v_min[1], min(self.twist.linear.y, self.v_max[1])))
            self.twist.angular.z = float(max(self.v_min[2], min(self.twist.angular.z, self.v_max[2])))


        self.pub_vel.publish(self.twist)
        self._last_time = now

        # visulize roll out 
        # states = self.mppi_controller.states.squeeze(0)[:50]
        # self.sampled_path(states,"odom")
        # local_path = self.mppi_controller.get_rollouts(self.current_state)
        # self.publish_path(local_path, self.local_path_publisher, "odom")


    # ========================
    # === MPPI Core Logic ===
    # ========================
    def non_holonomic_dynamics(self, state: torch.Tensor, action: torch.Tensor, delta_t: float = None) -> torch.Tensor:
        """
        Non-holonomic dynamics with acceleration-based control.

        - State:  [x, y, theta, vx, vy, omega]
            x, y     : position
            theta    : orientation (yaw)
            vx       : linear velocity (forward)
            vy       : lateral velocity (should remain 0 for non-holonomic robot)
            omega    : angular velocity (yaw rate)

        - Action: [acc, acc_omega]
            acc        : linear acceleration (forward)
            acc_omega  : angular acceleration (change in omega)

        - Returns:
            Updated state after applying the action over time step dt.
        """
        dt = delta_t or self._dt

        # Extract state variables
        x, y, theta = state[:, 0], state[:, 1], state[:, 2]
        vx, vy, omega = state[:, 3], state[:, 4], state[:, 5]

        # Clamp actions within bounds
        acc = torch.clamp(action[:, 0], self.acc_min[0], self.acc_max[0])
        acc_omega = torch.clamp(action[:, 1], self.acc_min[1], self.acc_max[1])

        # Normalize angle
        theta = self.angle_normalize(theta)

        # Update velocities
        _vx = vx + acc * dt
        _omega = omega + acc_omega * dt

        vx = torch.clamp(_vx, self.v_min[0], self.v_max[0])
        omega = torch.clamp(_omega, self.v_min[1], self.v_max[1])

        # Update positions and heading
        x = x + vx * torch.cos(theta) * dt
        y = y + vx * torch.sin(theta) * dt
        theta = self.angle_normalize(theta + omega * dt)

        # Set vy to zero due to non-holonomic constraint
        vy = torch.zeros_like(vx)

        return torch.stack([x, y, theta, vx, vy, omega], dim=1)
    

    def holonomic_dynamics(self, state: torch.Tensor, action: torch.Tensor, delta_t=None) -> torch.Tensor:
        """
        Holonomic dynamics with acceleration-based control.
        
        - State:  [x, y, theta, vx, vy, omega]
            x, y     : position
            theta    : orientation (yaw)
            vx, vy   : linear velocities (global frame)
            omega    : angular velocity

        - Action: [acc_x, acc_y, acc_omega]
            acc_x     : linear acceleration in x (global)
            acc_y     : linear acceleration in y (global)
            acc_omega : angular acceleration

        - Returns:
            Updated state after applying the action over time step dt.
        """

        dt = delta_t or self._dt

        # Extract state
        x, y, theta = state[:, 0], state[:, 1], state[:, 2]
        vx, vy, omega = state[:, 3], state[:, 4], state[:, 5]

        # Clamp actions
        acc_x = torch.clamp(action[:, 0], self.acc_min[0], self.acc_max[0])
        acc_y = torch.clamp(action[:, 1], self.acc_min[1], self.acc_max[1])
        acc_omega = torch.clamp(action[:, 2], self.acc_min[2], self.acc_max[2])

        # Update velocities
        _vx = vx + acc_x * dt
        _vy = vy + acc_y * dt
        _omega = omega + acc_omega * dt

        vx = torch.clamp(_vx, self.v_min[0], self.v_max[0])
        vy = torch.clamp(_vy, self.v_min[1], self.v_max[1])
        omega = torch.clamp(_omega, self.v_min[2], self.v_max[2])

        # Normalize angle
        theta = self.angle_normalize(theta)

        # Update position (world frame)
        x = x + (vx * torch.cos(theta) - vy * torch.sin(theta)) * dt
        y = y + (vx * torch.sin(theta) + vy * torch.cos(theta)) * dt
        theta = self.angle_normalize(theta + omega * dt)

        return torch.stack([x, y, theta, vx, vy, omega], dim=1)

    def nonholonomic_cost_function(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Goal tracking cost"""
        dist_cost = torch.sum((state[:, :2] - self._goal_pos[:2])**2, dim=1)
        theta_error = self.angle_normalize(state[:, 2] - self._goal_pos[2])
        angle_cost = theta_error**2  
        return self.distance_weight*dist_cost + self.angle_weight*angle_cost

    def holonomic_cost_function(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Goal tracking cost"""
         # Separate x and y distance errors
        dx = state[:, 0] - self._goal_pos[0]
        dy = state[:, 1] - self._goal_pos[1]
        
        # Compute squared errors separately
        dist_cost_x = dx**2
        dist_cost_y = dy**2

        theta_error = self.angle_normalize(state[:, 2] - self._goal_pos[2])
        angle_cost = theta_error**2  
        # Combine weighted cost
        return (
            self.distance_weight_x * dist_cost_x +
            self.distance_weight_y * dist_cost_y +
            self.angle_weight * angle_cost
        )
    
    def nonholonomic_terminal_cost(self, states, actions):
        return self.terminal_scale * torch.sum((states[..., -1, :2] - self._goal_pos[:2])**2, dim=-1)
    
    def holonomic_terminal_cost(self, states, actions):
        """
        Terminal cost that separates x and y components.
        
        - states: Tensor of shape [..., T, state_dim]
        - actions: Unused, but kept for interface compatibility
        """
        final_state = states[..., -1, :]  # shape: [..., state_dim]
        dx = final_state[..., 0] - self._goal_pos[0]
        dy = final_state[..., 1] - self._goal_pos[1]

        dist_cost_x = dx**2
        dist_cost_y = dy**2

        return (
            self.terminal_scale_x * dist_cost_x +
            self.terminal_scale_y * dist_cost_y
        )


    # ========================
    # === Utils ===
    # ========================
    def quaternion_to_yaw(self, quat: list) -> float:
        r = R.from_quat(quat)
        return r.as_euler('xyz')[2]

    def angle_normalize(self, x: torch.Tensor) -> torch.Tensor:
        return ((x + torch.pi) % (2 * torch.pi)) - torch.pi


# === Entry Point ===
def main(args=None):
    rclpy.init(args=args)
    node = MPPIController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
