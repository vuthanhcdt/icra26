import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Pose2D
from std_msgs.msg import Float32MultiArray


class MinDistancePublisher(Node):
    def __init__(self):
        super().__init__('min_distance_publisher')

        # Tạo subscriber cho min_distance và robot_human_pos
        self.min_distance_subscriber = self.create_subscription(
            Float32,
            'min_distance',
            self.min_distance_callback,
            10
        )
        self.robot_human_pos_subscriber = self.create_subscription(
            Pose2D,
            'robot_human_pos',
            self.robot_human_pos_callback,
            10
        )

        # Tạo publisher cho array float
        self.publisher = self.create_publisher(
            Float32MultiArray,
            'combined_data',
            10
        )

        self.min_distance_data = None
        self.robot_human_pos_data = None

    def min_distance_callback(self, msg):
        # Cập nhật giá trị min_distance
        self.min_distance_data = msg.data
        self.try_publish_combined_data()

    def robot_human_pos_callback(self, msg):
        # Cập nhật dữ liệu robot_human_pos
        self.robot_human_pos_data = [msg.x, msg.y, msg.theta]

    def try_publish_combined_data(self):
        # Chỉ publish khi đã nhận được cả min_distance và robot_human_pos
        if self.min_distance_data is not None and self.robot_human_pos_data is not None:
            combined_data = Float32MultiArray()
            combined_data.data = [self.min_distance_data] + self.robot_human_pos_data
            self.publisher.publish(combined_data)
            self.get_logger().info(f'Publishing combined data: {combined_data.data}')


def main(args=None):
    rclpy.init(args=args)

    node = MinDistancePublisher()

    rclpy.spin(node)

    # Khi không còn spin, cleanup
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
