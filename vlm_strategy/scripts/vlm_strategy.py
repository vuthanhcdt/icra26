import os
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from std_msgs.msg import String
from sensor_msgs.msg import Image
from openai import OpenAI
import cv2
from cv_bridge import CvBridge
import base64
from typing import Tuple
from visualization_msgs.msg import Marker

class VLMStrategy(Node):
    def __init__(self):
        super().__init__('vlm_strategy')

        # Parameters
        self.step_dist = self.declare_parameter('step_distance', 1.5).get_parameter_value().double_value  # meters
        self.goal_tolerance = self.declare_parameter('goal_tolerance', 0.2).get_parameter_value().double_value  # meters
 
        # Publishers
        self.prompt_pub = self.create_publisher(String, 'prompt_topic', 10)
        self.goal_pub = self.create_publisher(Pose, '/goal_pose', 10)
        self.marker_pub = self.create_publisher(Marker, '/human_marker', 10)

        # Subscribers
        self.image_sub = self.create_subscription(Image, '/realsense/image', self.image_callback, 10)
        self.human_pose_sub = self.create_subscription(Pose, '/human_local_pose', self.human_pose_callback, 10)

        # Timer
        self.timer = self.create_timer(1.0, self.timer_callback)  # 1 Hz

        # Internal state
        self.base64_image: str | None = None
        self.task_prompt: str = "human_position: Unknown"
        self.pending_action: str | None = None
        self.last_goal_pose: Pose | None = None
        self.human_pose: Pose | None = None

        # Prompt + OpenAI
        package_dir = os.path.dirname(os.path.dirname(__file__))
        self.instruction_prompt = self._read_prompt(os.path.join(package_dir, 'prompt', 'instruction_prompt.txt'))
        self.client = OpenAI()
        self.bridge = CvBridge()

    # ───────── Callbacks ─────────

    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            _, buffer = cv2.imencode('.jpg', cv_image)
            self.base64_image = base64.b64encode(buffer).decode('utf-8')
        except Exception as exc:
            self.get_logger().error(f"image_callback error: {exc}")

    def human_pose_callback(self, msg: Pose):
        self.human_pose = msg

        x, y = msg.position.x, msg.position.y
        pos_desc = 'Right' if abs(y) >= abs(x) and y > 0 else \
                   'Left' if abs(y) >= abs(x) and y <= 0 else \
                   'Front' if x > 0 else 'Back'
        self.task_prompt = f"human_position: {pos_desc}"
        if self.base64_image:
            if self.last_goal_pose is None:
                self.get_logger().info("First VLM call — no previous goal.")
                self._call_vlm()
            # else:
            #     print(self._is_near_goal(msg, self.last_goal_pose))
            #     print(self.last_goal_pose)
            elif self._is_near_goal(msg, self.last_goal_pose):
                self.get_logger().info("Reached goal — triggering VLM.")
                self._call_vlm()

    # ───────── Timer ─────────

    def timer_callback(self):
        if self.pending_action:
            goal_x, goal_y = self._action_to_offset(self.pending_action)
            pose = Pose()
            pose.position.x = goal_x
            pose.position.y = goal_y
            self.goal_pub.publish(pose)
            pose_human = Pose()
            pose_human.position.x = -goal_x
            pose_human.position.y = -goal_y
            self.last_goal_pose = pose_human
            # self.get_logger().info(f"Published goal: x={goal_x:.2f}, y={goal_y:.2f} ← from '{self.pending_action}'")
            self.pending_action = None

    # ───────── Internals ─────────

    def _call_vlm(self):
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.instruction_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.task_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self.base64_image}"}},
                        ],
                    },
                ],
                max_tokens=20,
            )
            action = response.choices[0].message.content.strip()
            self.pending_action = action
            self.prompt_pub.publish(String(data=action))
            self.get_logger().info(f"VLM → action: '{action}'")
            self.get_logger().info(self.task_prompt)
        except Exception as exc:
            self.get_logger().error(f"VLM call failed: {exc}")

    def _action_to_offset(self, action: str) -> Tuple[float, float]:
        a = action.lower()
        if 'right' in a:
            return -0.1, -self.step_dist
        if 'left' in a:
            return -0.1, self.step_dist
        if 'front' in a or 'forward' in a:
            return -self.step_dist, 0.0
        if 'back' in a or 'behind' in a:
            return self.step_dist, 0.0
        return 0.0, 0.0

    def _is_near_goal(self, p1: Pose, p2: Pose) -> bool:
        dx = p1.position.x - p2.position.x
        dy = p1.position.y - p2.position.y
        distance = math.sqrt(dx ** 2 + dy ** 2)
        # print(distance)
        return distance < self.goal_tolerance
    
   

        
    @staticmethod
    def _read_prompt(file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as fp:
                return fp.read().strip()
        except Exception as exc:
            rclpy.logging.get_logger('VLMStrategy').error(f"read_prompt error: {exc}")
            return ""

# ───────── Entry ─────────

def main(args=None):
    rclpy.init(args=args)
    node = VLMStrategy()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
