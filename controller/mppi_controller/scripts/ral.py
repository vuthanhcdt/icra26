import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist

class JoyToCmdVel(Node):
    def __init__(self):
        super().__init__('joy_to_cmd_vel')
        
        # Khởi tạo subscriber cho topic joy
        self.joy_sub = self.create_subscription(
            Joy,
            '/joy',
            self.joy_callback,
            10)
        
        # Khởi tạo publisher cho topic cmd_vel
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10)
        
        # Các tham số để điều chỉnh tốc độ
        self.linear_scale = 1.5  # Tỷ lệ tốc độ tuyến tính
        self.angular_scale = 1.0  # Tỷ lệ tốc độ góc
        
        # self.get_logger().info('Joy to CmdVel node started')

    def joy_callback(self, msg):
        # Tạo message Twist
        twist = Twist()
        # Gán giá trị tốc độ tuyến tính (x)
        twist.linear.x = float(msg.axes[1]) * self.linear_scale
        twist.linear.y = float(msg.axes[0]) * self.linear_scale
        twist.angular.z = float(msg.axes[2]) * self.angular_scale
        self.cmd_vel_pub.publish(twist)
        # self.get_logger().info(f'Publishing: linear.x={twist.linear.x:.2f}, linear.y={twist.linear.y:.2f}, angular.z={twist.angular.z:.2f}')

def main(args=None):
    rclpy.init(args=args)
    node = JoyToCmdVel()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down joy_to_cmd_vel node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()