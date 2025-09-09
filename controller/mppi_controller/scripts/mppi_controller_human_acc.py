#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, Point, Pose
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker, MarkerArray
from scipy.spatial.transform import Rotation as R
import torch
import numpy as np
from pytorch_mppi import mppi
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class MPPIConfig:
    """Configuration parameters for MPPI controller."""
    dtype: torch.dtype = torch.float32
    device: str = 'cuda'
    dt: float = 0.1
    horizon: int = 15
    num_samples: int = 1000
    lambda_: float = 0.01

    # Non-holonomic parameters
    non_holonomic_terminal_scale: float = 100.0
    non_holonomic_distance_weight: float = 100.0
    non_holonomic_angle_weight: float = 10.0

    # Holonomic parameters
    holonomic_terminal_scale_x: float = 300.0
    holonomic_terminal_scale_y: float = 200.0
    holonomic_distance_weight_x: float = 300.0
    holonomic_distance_weight_y: float = 200.0
    holonomic_angle_weight: float = 200.0

class Dynamics:
    """Handles robot dynamics for both holonomic and non-holonomic models."""
    
    def __init__(self, config: MPPIConfig, mode: str):
        self.config = config
        self.mode = mode
        self._setup_bounds()

    def _setup_bounds(self):
        """Configure control and velocity bounds based on mode."""
        if self.mode == "non_holonomic":
            self.acc_min = torch.tensor([-2.0, -2.0], device=self.config.device, dtype=self.config.dtype)
            self.acc_max = torch.tensor([2.0, 2.0], device=self.config.device, dtype=self.config.dtype)
            self.v_min = torch.tensor([-1.0, -1.0], device=self.config.device, dtype=self.config.dtype)
            self.v_max = torch.tensor([1.0, 1.0], device=self.config.device, dtype=self.config.dtype)
            self.noise_sigma = torch.diag(torch.tensor([2.0, 2.0], device=self.config.device, dtype=torch.double))
        elif self.mode == "holonomic":
            self.acc_min = torch.tensor([-1.0, -1.0, -1.0], device=self.config.device, dtype=self.config.dtype)
            self.acc_max = torch.tensor([1.0, 1.0, 1.0], device=self.config.device, dtype=self.config.dtype)
            self.v_min = torch.tensor([-1.0, -1.0, -1.0], device=self.config.device, dtype=self.config.dtype)
            self.v_max = torch.tensor([1.0, 1.0, 1.0], device=self.config.device, dtype=self.config.dtype)
            self.noise_sigma = torch.diag(torch.tensor([1.0, 1.0, 1.0], device=self.config.device, dtype=torch.double))
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def non_holonomic(self, state: torch.Tensor, action: torch.Tensor, delta_t: Optional[float] = None) -> torch.Tensor:
        """Non-holonomic dynamics with acceleration-based control."""
        dt = delta_t or self.config.dt
        x, y, theta, vx, vy, omega = state[:, 0], state[:, 1], state[:, 2], state[:, 3], state[:, 4], state[:, 5]
        
        acc = torch.clamp(action[:, 0], self.acc_min[0], self.acc_max[0])
        acc_omega = torch.clamp(action[:, 1], self.acc_min[1], self.acc_max[1])
        
        theta = self._angle_normalize(theta)
        vx = torch.clamp(vx + acc * dt, self.v_min[0], self.v_max[0])
        omega = torch.clamp(omega + acc_omega * dt, self.v_min[1], self.v_max[1])
        
        x = x + vx * torch.cos(theta) * dt
        y = y + vx * torch.sin(theta) * dt
        theta = self._angle_normalize(theta + omega * dt)
        vy = torch.zeros_like(vx)
        
        return torch.stack([x, y, theta, vx, vy, omega], dim=1)

    def holonomic(self, state: torch.Tensor, action: torch.Tensor, delta_t: Optional[float] = None) -> torch.Tensor:
        """Holonomic dynamics with acceleration-based control."""
        dt = delta_t or self.config.dt
        x, y, theta, vx, vy, omega = state[:, 0], state[:, 1], state[:, 2], state[:, 3], state[:, 4], state[:, 5]
        
        acc_x = torch.clamp(action[:, 0], self.acc_min[0], self.acc_max[0])
        acc_y = torch.clamp(action[:, 1], self.acc_min[1], self.acc_max[1])
        acc_omega = torch.clamp(action[:, 2], self.acc_min[2], self.acc_max[2])
        
        vx = torch.clamp(vx + acc_x * dt, self.v_min[0], self.v_max[0])
        vy = torch.clamp(vy + acc_y * dt, self.v_min[1], self.v_max[1])
        omega = torch.clamp(omega + acc_omega * dt, self.v_min[2], self.v_max[2])
        
        theta = self._angle_normalize(theta)
        x = x + (vx * torch.cos(theta) - vy * torch.sin(theta)) * dt
        y = y + (vx * torch.sin(theta) + vy * torch.cos(theta)) * dt
        theta = self._angle_normalize(theta + omega * dt)
        
        return torch.stack([x, y, theta, vx, vy, omega], dim=1)

    def _angle_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize angle to [-pi, pi]."""
        return ((x + torch.pi) % (2 * torch.pi)) - torch.pi

    def get_dynamics(self):
        """Return appropriate dynamics function based on mode."""
        return self.non_holonomic if self.mode == "non_holonomic" else self.holonomic

class CostFunctions:
    """Handles cost functions for MPPI optimization."""
    
    def __init__(self, config: MPPIConfig, mode: str, goal_pos: Optional[torch.Tensor] = None, 
                 human_pos: Optional[torch.Tensor] = None, safe_distance: float = 1.2, 
                 cbf_weight: float = 10000000.0, cbf_alpha: float = 0.1):
        self.config = config
        dynamics = Dynamics(config, mode)
        self.mode = mode
        self.dynamics = dynamics.get_dynamics()
        self.goal_pos = goal_pos
        self.human_pos = human_pos
        self.safe_distance = safe_distance
        self.cbf_weight = cbf_weight
        self.cbf_alpha = cbf_alpha


    def non_holonomic_running(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Non-holonomic running cost function."""
        dist_cost = torch.sum((state[:, :2] - self.goal_pos[:2])**2, dim=1)
        theta_error = self._angle_normalize(state[:, 2] - self.goal_pos[2])
        angle_cost = theta_error**2

        # CBF cost
        barrier_value = self.h_x(state)
        next_state = self.dynamics(state, action, self.config.dt)
        next_barrier_value = self.h_x(next_state)
        cbf_cost = -next_barrier_value + self.cbf_alpha * barrier_value
        cbf_cost = self.cbf_weight * torch.clamp(cbf_cost, 0.0, float('inf'))

        return (
            self.config.non_holonomic_distance_weight * dist_cost +
            self.config.non_holonomic_angle_weight * angle_cost + 
            cbf_cost
        )
    

    def h_x(self, state: torch.Tensor) -> torch.Tensor:
        """Barrier function: distance to human minus safe distance."""
        robot_pos = state[:, :2]
        human_pos = self.human_pos[:2] if self.human_pos is not None else torch.tensor([2.0, 2.0], device=self.config.device, dtype=self.config.dtype)
        return torch.norm(robot_pos - human_pos, dim=1) - self.safe_distance



    def holonomic_running(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Holonomic running cost function."""
        dx = state[:, 0] - self.goal_pos[0]
        dy = state[:, 1] - self.goal_pos[1]
        dist_cost_x = dx**2
        dist_cost_y = dy**2
        theta_error = self._angle_normalize(state[:, 2] - self.goal_pos[2])
        angle_cost = theta_error**2

        # CBF cost
        barrier_value = self.h_x(state)
        next_state = self.dynamics(state, action, self.config.dt)
        next_barrier_value = self.h_x(next_state)
        cbf_cost = -next_barrier_value + self.cbf_alpha * barrier_value
        cbf_cost = self.cbf_weight * torch.clamp(cbf_cost, 0.0, float('inf'))

        return (
            self.config.holonomic_distance_weight_x * dist_cost_x +
            self.config.holonomic_distance_weight_y * dist_cost_y +
            self.config.holonomic_angle_weight * angle_cost + 
            cbf_cost
        )

    def non_holonomic_terminal(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Non-holonomic terminal cost function."""
        return self.config.non_holonomic_terminal_scale * torch.sum(
            (states[..., -1, :2] - self.goal_pos[:2])**2, dim=-1
        )

    def holonomic_terminal(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Holonomic terminal cost function."""
        final_state = states[..., -1, :]
        dx = final_state[..., 0] - self.goal_pos[0]
        dy = final_state[..., 1] - self.goal_pos[1]
        return (
            self.config.holonomic_terminal_scale_x * dx**2 +
            self.config.holonomic_terminal_scale_y * dy**2
        )

    def _angle_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize angle to [-pi, pi]."""
        return ((x + torch.pi) % (2 * torch.pi)) - torch.pi

    def get_cost_functions(self) -> Tuple[callable, callable]:
        """Return appropriate cost functions based on mode."""
        if self.mode == "non_holonomic":
            return self.non_holonomic_running, self.non_holonomic_terminal
        return self.holonomic_running, self.holonomic_terminal

class MPPIController(Node):
    """ROS 2 node implementing MPPI control for robot navigation."""
    
    def __init__(self, mode: str = "holonomic"):
        super().__init__('mppi_controller')

        # Declare parameters from YAML
        self.declare_parameter('topic_velocity', 'cmd_vel')
        self.declare_parameter('dt', 0.1)
        self.declare_parameter('horizon', 15)
        self.declare_parameter('num_samples', 1000)
        self.declare_parameter('lambda', 0.01)
        self.declare_parameter('non_holonomic_terminal_scale', 100.0)
        self.declare_parameter('non_holonomic_distance_weight', 100.0)
        self.declare_parameter('non_holonomic_angle_weight', 10.0)
        self.declare_parameter('holonomic_terminal_scale_x', 300.0)
        self.declare_parameter('holonomic_terminal_scale_y', 200.0)
        self.declare_parameter('holonomic_distance_weight_x', 300.0)
        self.declare_parameter('holonomic_distance_weight_y', 200.0)
        self.declare_parameter('holonomic_angle_weight', 200.0)
        self.declare_parameter('safe_distance', 1.2)
        self.declare_parameter('cbf_weight', 10000000.0)
        self.declare_parameter('cbf_alpha', 0.1)
        self.declare_parameter('local_x', 0.3)
        self.declare_parameter('local_y', 1.0)
        
        # Initialize MPPIConfig with parameters from YAML
        self.config = MPPIConfig(
            dtype=torch.float32,
            device='cuda',
            dt=self.get_parameter('dt').get_parameter_value().double_value,
            horizon=self.get_parameter('horizon').get_parameter_value().integer_value,
            num_samples=self.get_parameter('num_samples').get_parameter_value().integer_value,
            lambda_=self.get_parameter('lambda').get_parameter_value().double_value,
            non_holonomic_terminal_scale=self.get_parameter('non_holonomic_terminal_scale').get_parameter_value().double_value,
            non_holonomic_distance_weight=self.get_parameter('non_holonomic_distance_weight').get_parameter_value().double_value,
            non_holonomic_angle_weight=self.get_parameter('non_holonomic_angle_weight').get_parameter_value().double_value,
            holonomic_terminal_scale_x=self.get_parameter('holonomic_terminal_scale_x').get_parameter_value().double_value,
            holonomic_terminal_scale_y=self.get_parameter('holonomic_terminal_scale_y').get_parameter_value().double_value,
            holonomic_distance_weight_x=self.get_parameter('holonomic_distance_weight_x').get_parameter_value().double_value,
            holonomic_distance_weight_y=self.get_parameter('holonomic_distance_weight_y').get_parameter_value().double_value,
            holonomic_angle_weight=self.get_parameter('holonomic_angle_weight').get_parameter_value().double_value
        )

        self.mode = mode
        self.current_state = torch.zeros(6, dtype=self.config.dtype, device=self.config.device)
        self.safe_distance = self.get_parameter('safe_distance').get_parameter_value().double_value
        self.goal_pos = None
        self.human_pos = None
        self.local_x = self.get_parameter('local_x').get_parameter_value().double_value
        self.local_y = self.get_parameter('local_y').get_parameter_value().double_value
        self.cbf_weight = self.get_parameter('cbf_weight').get_parameter_value().double_value
        self.cbf_alpha = self.get_parameter('cbf_alpha').get_parameter_value().double_value
        self.twist = Twist()
        self.marker_radius: float = self.declare_parameter('marker_radius', 0.5).value
        self.marker_height: float = self.declare_parameter('marker_height', 0.01).value
        self.last_time = self.get_clock().now()

        
        self._setup_mppi()
        self._setup_ros_interfaces()

    def _setup_mppi(self):
        """Initialize MPPI controller."""
        dynamics = Dynamics(self.config, self.mode)
      
        cost = CostFunctions(
            self.config, 
            self.mode, 
            self.goal_pos, 
            self.human_pos,
            safe_distance=self.get_parameter('safe_distance').get_parameter_value().double_value,
            cbf_weight=self.get_parameter('cbf_weight').get_parameter_value().double_value,
            cbf_alpha=self.get_parameter('cbf_alpha').get_parameter_value().double_value
        )

        running_cost, terminal_cost = cost.get_cost_functions()
        
        self.mppi_controller = mppi.MPPI(
            dynamics=dynamics.get_dynamics(),
            running_cost=running_cost,
            terminal_state_cost=terminal_cost,
            nx=6,
            noise_sigma=dynamics.noise_sigma,
            num_samples=self.config.num_samples,
            horizon=self.config.horizon,
            lambda_=self.config.lambda_,
            u_min=dynamics.acc_min,
            u_max=dynamics.acc_max,
            device=self.config.device
        )

    def _setup_ros_interfaces(self):
        """Setup ROS publishers and subscribers."""
        topic_vel = self.get_parameter('topic_velocity').get_parameter_value().string_value
        
        self.pub_vel = self.create_publisher(Twist, topic_vel, 2)
        self.local_path_publisher = self.create_publisher(Path, '/local_path', 1)
        self.sampled_path_publisher = self.create_publisher(MarkerArray, '/sampled_path', 1)
        
        self.odom_sub = self.create_subscription(Odometry, '/odom', self._odom_callback, 10)
        self.marker_pub = self.create_publisher(Marker, '/human_marker', 10)
        self.human_sub = self.create_subscription(Pose, '/human_global_pose', self._human_callback, 10)
        self.goal_sub = self.create_subscription(Pose, '/goal_pose', self._goal_local_callback, 10)
        self.timer = self.create_timer(0.001, self._timer_callback)

    def _odom_callback(self, msg: Odometry):
        """Update current state from odometry data."""
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        twist = msg.twist.twist
        yaw = self._quaternion_to_yaw([ori.x, ori.y, ori.z, ori.w])
        self.current_state = torch.tensor(
            [pos.x, pos.y, yaw, twist.linear.x, twist.linear.y, twist.angular.z],
            dtype=self.config.dtype,
            device=self.config.device
        )

    def _human_callback(self, msg: Pose):
        """Update goal position based on human pose."""
        # self._publish_cylinder_marker(msg)
        self.human_x = msg.position.x
        self.human_y = msg.position.y
        self.human_theta = self._quaternion_to_yaw([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])

        self.goal_x = self.human_x + self.local_x * np.cos(self.human_theta) - self.local_y * np.sin(self.human_theta)
        self.goal_y = self.human_y + self.local_x * np.sin(self.human_theta) + self.local_y * np.cos(self.human_theta)
        self.goal_pos = torch.tensor([self.goal_x, self.goal_y, self.human_theta], device=self.config.device, dtype=self.config.dtype)
        self.human_pos = torch.tensor([self.human_x, self.human_y, self.human_theta], device=self.config.device, dtype=self.config.dtype)

        # print( self.goal_x, self.goal_y)
        
        # Update cost functions with new goal position
        self.mppi_controller.running_cost = CostFunctions(self.config, self.mode, self.goal_pos, self.human_pos, self.safe_distance, self.cbf_weight, self.cbf_alpha).get_cost_functions()[0]
        self.mppi_controller.terminal_state_cost = CostFunctions(self.config, self.mode, self.goal_pos, self.human_pos, self.safe_distance, self.cbf_weight, self.cbf_alpha).get_cost_functions()[1]


    def _goal_local_callback(self, msg: Pose):
        """Update goal position based on human pose from VLM/LLM."""
        self.local_x = msg.position.x
        self.local_y = msg.position.y


    def _timer_callback(self):
        """Main control loop."""
        if self.goal_pos is None:
            return
            
        action = self.mppi_controller.command(self.current_state)
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9
        
        if self.mode == "non_holonomic":
            self.twist.linear.x += float(action[0]) * dt
            self.twist.angular.z += float(action[1]) * dt
            self.twist.linear.x = float(max(self.mppi_controller.u_min[0], min(self.twist.linear.x, self.mppi_controller.u_max[0])))
            self.twist.angular.z = float(max(self.mppi_controller.u_min[1], min(self.twist.angular.z, self.mppi_controller.u_max[1])))
        else:
            self.twist.linear.x += float(action[0]) * dt
            self.twist.linear.y += float(action[1]) * dt
            self.twist.angular.z += float(action[2]) * dt
            self.twist.linear.x = float(max(self.mppi_controller.u_min[0], min(self.twist.linear.x, self.mppi_controller.u_max[0])))
            self.twist.linear.y = float(max(self.mppi_controller.u_min[1], min(self.twist.linear.y, self.mppi_controller.u_max[1])))
            self.twist.angular.z = float(max(self.mppi_controller.u_min[2], min(self.twist.angular.z, self.mppi_controller.u_max[2])))
        
        self.pub_vel.publish(self.twist)
        # local_path = self.mppi_controller.get_rollouts(self.current_state)
        # self.publish_path(local_path, self.local_path_publisher, "odom")
        self.last_time = now


     # —— Marker ——
    def _publish_cylinder_marker(self, pose: Pose):
        """Publish a CUBE marker around the human to represent a flat circle."""
        marker = Marker()
        marker.header.frame_id = "odom"
        # marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'human_cylinder'
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD

        marker.pose.position.x = pose.position.x
        marker.pose.position.y = pose.position.y
        marker.pose.position.z = 0.01  # Flat on the ground
        marker.pose.orientation.w = 1.0

        marker.scale.x = self.marker_radius * 2.0
        marker.scale.y = self.marker_radius * 2.0
        marker.scale.z = self.marker_height

        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.6

        self.marker_pub.publish(marker)

        
    def publish_path(self, rollout: torch.Tensor, publisher: rclpy.publisher.Publisher, base: str):
        """Publish a single rollout path as nav_msgs/Path."""
        path_msg = Path()
        path_msg.header.frame_id = base
        path_msg.header.stamp = self.get_clock().now().to_msg()
        
        for t in range(rollout.shape[1]):
            pose = PoseStamped()
            pose.header.frame_id = base
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = float(rollout[0, t, 0].cpu())
            pose.pose.position.y = float(rollout[0, t, 1].cpu())
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        
        publisher.publish(path_msg)

    def publish_sampled_paths(self, state: torch.Tensor, base: str):
        """Publish sampled paths for visualization."""
        marker_array = MarkerArray()
        for rollout_idx in range(state.shape[0]):
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
            marker.color.a = 1.0
            marker.lifetime = rclpy.duration.Duration(seconds=1.0).to_msg()
            
            for t in range(state.shape[1]):
                pt = Point()
                pt.x = float(state[rollout_idx, t, 0].cpu())
                pt.y = float(state[rollout_idx, t, 1].cpu())
                pt.z = 0.0
                marker.points.append(pt)
            
            marker_array.markers.append(marker)
        
        self.sampled_path_publisher.publish(marker_array)

    def _quaternion_to_yaw(self, quat: list) -> float:
        """Convert quaternion to yaw angle."""
        return R.from_quat(quat).as_euler('xyz')[2]

def main(args=None):
    """Entry point for the MPPI controller node."""
    rclpy.init(args=args)
    node = MPPIController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()