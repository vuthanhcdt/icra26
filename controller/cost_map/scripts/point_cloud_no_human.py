#!/usr/bin/env python3
import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import Pose2D
import sensor_msgs_py.point_cloud2 as pc2

class FilterPointCloud(Node):
    def __init__(self):
        super().__init__('filter_point_cloud')

        # Subscription to PointCloud2 topic
        self.pointcloud_sub = self.create_subscription(
            PointCloud2, '/velodyne_points', self.pointcloud_callback, 10)

        # Subscription to human position topic
        self.human_sub = self.create_subscription(
            Pose2D, '/human_robot_pos', self.human_callback, 10)

        # Publisher for filtered PointCloud2
        self.filtered_cloud_pub = self.create_publisher(PointCloud2, '/filtered_points', 10)

        # Parameters
        self.human_radian = self.declare_parameter('human_radian', 1.0).get_parameter_value().double_value

        # Initialize variables for storing data
        self.human_position = None
        self.pointcloud_data = None

        # Timer for processing data
        self.timer = self.create_timer(0.01, self.process_data)  # 0.01s interval (100 Hz)

    def human_callback(self, msg):
        """Callback to update the human position."""
        self.human_position = np.array([msg.x, msg.y], dtype=np.float32)

    def pointcloud_callback(self, msg):
        """Callback to store PointCloud2 data."""
        self.pointcloud_data = msg

    def process_data(self):
        """Timer-based method to process stored PointCloud2 data."""
        if self.human_position is None or self.pointcloud_data is None:
            return

        human_position = self.human_position
        radius_squared = self.human_radian ** 2  # Human radius squared
        max_radius_squared = 10.0 ** 2  # Maximum allowed radius squared (10 meters)

        try:
            # Read points as a generator
            points = pc2.read_points(self.pointcloud_data, field_names=('x', 'y', 'z'), skip_nans=True)
            # Extract points explicitly and convert to numpy array
            cloud_points = np.array([[p[0], p[1], p[2]] for p in points], dtype=np.float32)
        except Exception as e:
            self.get_logger().error(f"Error reading PointCloud data: {e}")
            return

        if cloud_points.shape[1] != 3:
            self.get_logger().error("PointCloud data is not in the expected format (Nx3). Skipping.")
            return

        # Vectorized filtering for two conditions
        distances_squared = np.sum((cloud_points[:, :2] - human_position) ** 2, axis=1)
        filtered_points = cloud_points[(distances_squared > radius_squared) & (distances_squared <= max_radius_squared)]

        # Publish filtered points
        self.filtered_cloud_pub.publish(self.create_pointcloud(self.pointcloud_data.header, filtered_points))

    def create_pointcloud(self, header, points):
        """Create a PointCloud2 message from filtered points."""
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        return pc2.create_cloud(header, fields, points)

def main(args=None):
    rclpy.init(args=args)
    pointcloud_filter = FilterPointCloud()
    rclpy.spin(pointcloud_filter)
    pointcloud_filter.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
