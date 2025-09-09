#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose,  Point, Quaternion
from std_msgs.msg import Bool
import math
import ast  # Safe string-to-python list conversion
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np

class ActorControl(Node):
    def __init__(self):
        super().__init__('actor_control')
        self.position_control_mode = self.declare_parameter('position_control_mode',True).get_parameter_value().bool_value
        self.topic_actor_velocity = self.declare_parameter('topic_actor_velocity', 'cmd_vel_actor').value
        self.topic_actor_position = self.declare_parameter('topic_actor_position', 'pose_control_actor').value
        self.topic_actor_position_mode = self.declare_parameter('topic_actor_position_mode', 'mode_position_actor').value
        self.topic_actor_global_pose = self.declare_parameter('topic_actor_global_pose', 'human_global_pose').value
        self.topic_actor_local_pose = self.declare_parameter('topic_actor_local_pose', 'human_local_pose').value
        self.topic_ped_global_pose = self.declare_parameter('topic_ped_global_pose', 'ped_global_pose').value
        self.topic_ped_local_pose = self.declare_parameter('topic_ped_local_pose', 'ped_local_pose').value
        self.scenario = self.declare_parameter('scenario', 's1').value
        self.speed = self.declare_parameter('speed',0.8).get_parameter_value().double_value
        self.goal_x_1 = self.declare_parameter('goal_x_1',0.3).get_parameter_value().double_value
        self.goal_y_1 = self.declare_parameter('goal_y_1',0.9).get_parameter_value().double_value
        self.goal_x_2 = self.declare_parameter('goal_x_2',0.3).get_parameter_value().double_value
        self.goal_y_2 = self.declare_parameter('goal_y_2',0.9).get_parameter_value().double_value
        self.timer = self.create_timer(0.05, self.timer_callback)
        self.actor_velocity_control = self.create_publisher(Twist, self.topic_actor_velocity, 3)
        self.actor_position_control = self.create_publisher(Pose, self.topic_actor_position, 3)
        self.actor_position_mode = self.create_publisher(Bool,  self.topic_actor_position_mode, 3)
        self.get_ped = False
        self.marker_radius: float = self.declare_parameter('marker_radius', 0.5).value
        self.marker_height: float = self.declare_parameter('marker_height', 0.01).value
        self.marker_pub = self.create_publisher(Marker, '/human_marker', 10)
        self.semi_major_axis = self.declare_parameter('semi_major_axis', 1.1).get_parameter_value().double_value
        self.semi_minor_axis = self.declare_parameter('semi_minor_axis', 0.9).get_parameter_value().double_value
        self.num_points = self.declare_parameter('num_points',8).get_parameter_value().integer_value
        self.interactive_space = self.create_publisher(MarkerArray, 'interactive_space', 3)
        self.x_vals, self.y_vals = self.generate_ellipse_points(self.semi_major_axis, self.semi_minor_axis, self.num_points)

        # print(self.x_vals, self.y_vals)
       

        self.mode = Bool()
        self.mode.data=self.position_control_mode

        # Subscriber đến /human_global_pose với message type Pose
       
        self.human_local_pose_subscriber = self.create_subscription(
            Pose,
            self.topic_actor_local_pose,
            self.human_local_pose_callback,
            5
        )

               
        self.human_pose_subscriber = self.create_subscription(
            Pose,
            self.topic_actor_global_pose,
            self.human_pose_callback,
            5
        )


        self.ped_pose_subscriber = self.create_subscription(
            Pose,
            self.topic_ped_global_pose,
            self.ped_pose_callback,
            5
        )


        if self.topic_actor_velocity == "cmd_vel_ped":
            self.human_local_pose_subscriber = self.create_subscription(
                Pose,
                self.topic_ped_local_pose,
                self.ped_local_pose_callback,
                5
            )
            self.goal_pub = self.create_publisher(Pose, '/goal_pose', 10)


        # Waypoints - danh sách các vị trí Pose
        self.declare_parameter('waypoints', '[[0.0, 0.0, 0.0]]')  # 🔹 Thêm dòng này
        self.waypoints = self.create_waypoints()
        self.current_index = 0  # index waypoint hiện tại
        self.current_human_pose = None  # lưu pose hiện tại nhận được từ human
        self.current_ped_pose = None
        self.current_local_human_pose = None
        self.dist_to_robot = None



    def create_waypoints(self):
        waypoint_str = self.get_parameter('waypoints').get_parameter_value().string_value

        try:
            waypoint_array = ast.literal_eval(waypoint_str)  # safely parse string to list
        except Exception as e:
            self.get_logger().error(f"Failed to parse waypoints string: {e}")
            return []

        waypoints = []
        for coords in waypoint_array:
            if len(coords) != 3:
                self.get_logger().warn(f"Skipping invalid waypoint: {coords}")
                continue
            pose = Pose()
            pose.position = Point(x=coords[0], y=coords[1], z=coords[2])
            pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            waypoints.append(pose)

        self.get_logger().info(f"Loaded {len(waypoints)} waypoints from parameters.")
        return waypoints
    

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

        marker.color.r = 0.1
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 0.9

        self.marker_pub.publish(marker)

     

    def generate_ellipse_points(self, a, b, n):
        """Generates n points on an ellipse with semi-major axis a and semi-minor axis b."""
        theta = np.linspace(0, 2 * np.pi, n)
        return a * np.cos(theta), b * np.sin(theta)
    

    def human_pose_callback(self, msg: Pose):
        self.current_human_pose = msg
        if self.topic_actor_velocity == "cmd_vel_ped":
            return
        self._publish_cylinder_marker(msg)
        # Tạo MarkerArray từ ellipse quanh người
        marker_array = MarkerArray()
        for i, (x, y) in enumerate(zip(self.x_vals, self.y_vals)):
            marker = Marker()
            marker.header.frame_id = "human_local_link"
            # marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "ellipse"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = 0.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = 1.0
            marker_array.markers.append(marker)

        # Publish MarkerArray
        self.interactive_space.publish(marker_array)

    
    def human_local_pose_callback(self, msg: Pose):
        self.current_local_human_pose = msg

    def ped_pose_callback(self, msg: Pose):
        self.current_global_ped_pose = msg

    def ped_local_pose_callback(self, msg: Pose):
        self.current_ped_pose = msg
        ped_x = msg.position.x
        ped_y = msg.position.y
        self.dist_to_robot = math.sqrt((ped_x)**2 + (ped_y)**2)
        if  self.scenario =="s1":
            if self.dist_to_robot < 3.6 and self.get_ped==False:
                goal_pose = Pose()
                goal_pose.position = Point(x=self.goal_x_1, y=self.goal_y_1, z=0.0)
                goal_pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                self.goal_pub.publish(goal_pose)
                self.get_ped = True
            elif ped_x < -0.5 and self.get_ped: 
                goal_pose = Pose()
                goal_pose.position = Point(x=self.goal_x_2, y=self.goal_y_2, z=0.0)
                goal_pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                self.goal_pub.publish(goal_pose)
        elif self.scenario =="s2":
            if self.current_local_human_pose is None:
                return
            if self.dist_to_robot < 3.6 and self.get_ped==False:
                goal_pose = Pose()
                goal_pose.position = Point(x=self.goal_x_1, y=self.goal_y_1, z=0.0)
                goal_pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                self.goal_pub.publish(goal_pose)
                self.get_ped = True
            elif self.current_local_human_pose.position.x > 0.1 and self.get_ped: 
                goal_pose = Pose()
                goal_pose.position = Point(x=self.goal_x_2, y=self.goal_y_2, z=0.0)
                goal_pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                self.goal_pub.publish(goal_pose)
        elif self.scenario =="s4":
            if self.current_local_human_pose is None:
                return
            if self.dist_to_robot < 3.7 and self.get_ped==False:
                goal_pose = Pose()
                goal_pose.position = Point(x=self.goal_x_1, y=self.goal_y_1, z=0.0)
                goal_pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                self.goal_pub.publish(goal_pose)
                self.get_ped = True
            elif self.dist_to_robot < 2.0 and self.get_ped: 
                goal_pose = Pose()
                goal_pose.position = Point(x=self.goal_x_2, y=self.goal_y_2, z=0.0)
                goal_pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                self.goal_pub.publish(goal_pose)
        elif self.scenario =="s5":
            if self.current_local_human_pose is None:
                return
            if self.dist_to_robot < 2.5 and self.get_ped==False:
                goal_pose = Pose()
                goal_pose.position = Point(x=self.goal_x_1, y=self.goal_y_1, z=0.0)
                goal_pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                self.goal_pub.publish(goal_pose)
                self.get_ped = True
            elif self.dist_to_robot < 1.0 and self.get_ped: 
                goal_pose = Pose()
                goal_pose.position = Point(x=self.goal_x_2, y=self.goal_y_2, z=0.0)
                goal_pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                self.goal_pub.publish(goal_pose)
                


    def distance(self, p1: Point, p2: Point) -> float:
        return math.sqrt(
            (p1.x - p2.x) ** 2 +
            (p1.y - p2.y) ** 2 +
            (p1.z - p2.z) ** 2
        )

    def timer_callback(self):

        if self.current_human_pose is None and self.current_ped_pose is None:
            self.get_logger().info("Waiting for human pose...")
            return

        self.mode.data=self.position_control_mode
        self.actor_position_mode.publish(self.mode)

        target_pose = self.waypoints[self.current_index]

        if self.topic_actor_velocity == "cmd_vel_actor":
            dist = self.distance(self.current_human_pose.position, target_pose.position)
        else:
            dist = self.distance(self.current_global_ped_pose.position, target_pose.position)
        # self.get_logger().info(f"Distance to waypoint {self.current_index}: {dist:.3f}")

        if dist < 0.05:
            if self.current_index < len(self.waypoints) - 1:
                self.current_index += 1
                # self.get_logger().info(f"Switching to waypoint {self.current_index}")
            else:
                # self.get_logger().info("Reached final waypoint. Stopping.")
                return  
        if self.scenario =="s5" and self.topic_actor_velocity != "cmd_vel_actor":
            if self.dist_to_robot is not None and self.dist_to_robot<3.0:
                self.position_control_mode = True
            if self.position_control_mode:    
                self.actor_position_control.publish(target_pose)
                twist = Twist()
                twist.linear.x = self.speed 
                self.actor_velocity_control.publish(twist)
        else:
            self.actor_position_control.publish(target_pose)
            twist = Twist()
            twist.linear.x = self.speed 
            self.actor_velocity_control.publish(twist)


        
def main(args=None):
    rclpy.init(args=args)
    node = ActorControl()
    rclpy.spin(node)

if __name__ == '__main__':
    main()