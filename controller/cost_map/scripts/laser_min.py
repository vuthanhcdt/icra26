#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32

class MinDistanceNode(Node):
    def __init__(self):
        super().__init__('min_distance_node')
        
        # Subscriber nhận dữ liệu từ topic '/scan'
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.listener_callback,
            10)
        
        # Publisher để xuất kết quả ra topic '/min_distance'
        self.publisher = self.create_publisher(Float32, '/min_distance', 10)
        
        # Biến lưu trữ khoảng cách nhỏ nhất
        self.min_distance = Float32()
        
    def listener_callback(self, msg):
        # Lấy tất cả các khoảng cách từ dữ liệu scan
        distances = msg.ranges
        
        # Tìm khoảng cách nhỏ nhất (bỏ qua giá trị vô hạn)
        valid_distances = [d for d in distances if d != float('Inf') and d > 0.0 and d < 5.0]
        
        if valid_distances:
            self.min_distance.data = min(valid_distances)
            # Publish khoảng cách nhỏ nhất ra topic '/min_distance'
            self.publisher.publish(self.min_distance)
        # self.get_logger().info(f'Min Distance: {self.min_distance.data}')
        else:
            self.min_distance.data = float('Inf')  # Nếu không có khoảng cách hợp lệ, xuất Inf
            
        

def main(args=None):
    rclpy.init(args=args)
    node = MinDistanceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
