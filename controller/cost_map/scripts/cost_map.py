#!/usr/bin/env python3
import rclpy, math
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data

class LaserToOccupancyGrid(Node):
    def __init__(self):
        super().__init__('laser_to_occupancy_grid')
        self.subscription = self.create_subscription(LaserScan, 'scan', self.laser_callback, qos_profile_sensor_data)
        self.subscription  # prevent unused variable warning
        
        # Declare and initialize parameters
        self.map_resolution = self.declare_parameter('map_resolution', 0.2).get_parameter_value().double_value
        self.laser_max_dis = self.declare_parameter('laser_max_dis', 5.0).get_parameter_value().double_value
        self.inflation_offset = self.declare_parameter('inflation_offset', 1.0).get_parameter_value().double_value
        self.obstacle_offset = self.declare_parameter('obstacle_offset', 0.2).get_parameter_value().double_value
        self.obstacle_safe_dis = self.declare_parameter('obstacle_safe_dis', 0.3).get_parameter_value().double_value  # Declare this parameter
        self.human_radian = self.declare_parameter('human_radian', 0.75).get_parameter_value().double_value  # Declare this parameter

        self.human_sub = self.create_subscription(Odometry,'/human_robot_pos', self.human_callback,5)

        self.map_size_x = int(self.laser_max_dis * 2 / self.map_resolution)
        self.map_size_y = int(self.laser_max_dis * 2 / self.map_resolution)
        self.map_origin_x = -self.laser_max_dis
        self.map_origin_y = -self.laser_max_dis
        self.obstacle_size = round(self.obstacle_safe_dis / self.map_resolution)

        # Two layers for the map
        self.obs_layer = np.zeros((self.map_size_x, self.map_size_y), dtype=np.int16)  # Obstacle layer
        self.inflation_layer = np.zeros((self.map_size_x, self.map_size_y), dtype=np.int16)  # Inflation layer

        self.map_publisher = self.create_publisher(OccupancyGrid, 'local_costmap/costmap', 1)
        self.human_position=None

    
    def human_callback(self, msg):
        self.human_position=(msg.pose.pose.position.x,msg.pose.pose.position.y)


    def add_inflation_layer(self, x, y):
        """
        Add an inflation layer around the obstacle located at (x, y).
        Inflation is calculated using a Gaussian-like decay.
        """
        for dx in range(-self.obstacle_size, self.obstacle_size + 1):
            for dy in range(-self.obstacle_size, self.obstacle_size + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.map_size_x and 0 <= ny < self.map_size_y:
                    # Calculate the distance from the center obstacle
                    distance = math.sqrt(dx**2 + dy**2) * self.map_resolution
                    if distance <= self.obstacle_safe_dis:
                        # Apply a cost that decreases with distance
                        cost = int(100 * (1 - distance / self.obstacle_safe_dis))
                        self.inflation_layer[nx, ny] = max(self.inflation_layer[nx, ny], cost)

    def mark_obstacle_with_offset(self, x, y):
        """
        Mark obstacle layer with an offset area around the detected obstacle.
        """
        offset_size = round(self.obstacle_offset / self.map_resolution)  # Convert offset to map grid units
        for dx in range(-offset_size, offset_size + 1):
            for dy in range(-offset_size, offset_size + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.map_size_x and 0 <= ny < self.map_size_y:
                    self.obs_layer[nx, ny] = 100  # Mark obstacle area

    def laser_callback(self, msg: LaserScan):
        if self.human_position is None:
            self.get_logger().info("Please open the human's position node")
            return
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment
        ranges = msg.ranges

        # Reset map layers
        self.obs_layer.fill(0)
        self.inflation_layer.fill(0)

        for i, range_value in enumerate(ranges):
            if range_value < self.laser_max_dis:
                x_tmp = range_value * math.sin(angle_min + i * angle_increment)
                y_tmp = range_value * math.cos(angle_min + i * angle_increment)

                distance_to_human = math.sqrt((y_tmp - self.human_position[0])**2 + (x_tmp -  self.human_position[1])**2)
                if distance_to_human < self.human_radian:  
                    continue

                # Convert to map grid coordinates
                x = int((x_tmp - self.map_origin_x) / self.map_resolution)
                y = int((y_tmp - self.map_origin_y) / self.map_resolution)

                # Mark obstacle with offset in the obstacle layer
                self.mark_obstacle_with_offset(x, y)

                # Add inflation layer
                self.add_inflation_layer(x, y)

        # Combine the layers: obstacle layer and inflation layer
        # Use maximum of both layers for each cell
        self.map_data = np.maximum(self.obs_layer, self.inflation_layer)

        # Publish occupancy grid
        occupancy_grid_msg = OccupancyGrid()
        occupancy_grid_msg.header.frame_id = "base_scan"
        occupancy_grid_msg.info.resolution = self.map_resolution
        occupancy_grid_msg.info.width = self.map_size_x
        occupancy_grid_msg.info.height = self.map_size_y
        occupancy_grid_msg.info.origin.position.x = self.map_origin_x
        occupancy_grid_msg.info.origin.position.y = self.map_origin_y
        occupancy_grid_msg.data = self.map_data.flatten().tolist()  # Should be between -128 and 127
        self.map_publisher.publish(occupancy_grid_msg)

def main(args=None):
    rclpy.init(args=args)

    laser_to_occupancy_grid = LaserToOccupancyGrid()

    rclpy.spin(laser_to_occupancy_grid)

    laser_to_occupancy_grid.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
