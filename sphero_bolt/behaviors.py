#!/usr/bin/env python3
"""
Sphero BOLT — ROS2 named-behavior node.

Connects to the BOLT over BLE (spherov2 + bleak), holds the connection open,
and runs named behaviors sent as std_msgs/String on /bolt/behavior.

Run (ROS2 sourced, BOLT shaken awake, on the Pi):
    python3 bolt_node.py
Then publish commands from another terminal, e.g.:
    ros2 topic pub --once /bolt/behavior std_msgs/msg/String "{data: 'forward'}"
    ros2 topic pub --once /bolt/behavior std_msgs/msg/String "{data: 'led:blue'}"
    ros2 topic pub --once /bolt/behavior std_msgs/msg/String "{data: 'greet'}"
Ctrl-C stops the node and turns the matrix off.
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from spherov2 import scanner
from spherov2.sphero_edu import SpheroEduAPI
from spherov2.types import Color

BOLT_NAME = "SB-D2F0"          # your ball
TOPIC = "/bolt/behavior"

COLORS = {
    "red":    Color(255, 0, 0),
    "green":  Color(0, 255, 0),
    "blue":   Color(0, 0, 255),
    "yellow": Color(255, 255, 0),
    "purple": Color(160, 0, 255),
    "white":  Color(255, 255, 255),
    "off":    Color(0, 0, 0),
}


class BoltNode(Node):
    def __init__(self, bolt):
        super().__init__("sphero_bolt")
        self.bolt = bolt
        self.speed = 100          # 0-255
        self.roll_time = 2.0      # seconds

        # heading is relative to the ball's aim; calibrate later if needed.
        self.behaviors = {
            "forward":  lambda: self.bolt.roll(0,   self.speed, self.roll_time),
            "backward": lambda: self.bolt.roll(180, self.speed, self.roll_time),
            "right":    lambda: self.bolt.roll(90,  self.speed, self.roll_time),
            "left":     lambda: self.bolt.roll(270, self.speed, self.roll_time),
            "spin":     lambda: self.bolt.spin(360, 2),
            "stop":     lambda: self.bolt.stop_roll(),
            # expressive HRI combos (color + motion):
            "greet":    self._greet,
            "curious":  self._curious,
            "alarm":    self._alarm,
        }

        self.create_subscription(String, TOPIC, self.on_cmd, 10)
        self.get_logger().info(
            f"BOLT ready. Publish behavior names to {TOPIC}. "
            f"Known: {', '.join(self.behaviors)}, led:<color>")

    def on_cmd(self, msg):
        cmd = msg.data.strip().lower()
        self.get_logger().info(f"-> {cmd}")
        try:
            if cmd.startswith("led:"):
                color = COLORS.get(cmd.split(":", 1)[1])
                if color is None:
                    self.get_logger().warn(f"unknown color in '{cmd}'")
                    return
                self.bolt.set_main_led(color)
            elif cmd in self.behaviors:
                self.behaviors[cmd]()
            else:
                self.get_logger().warn(f"unknown behavior: '{cmd}'")
        except Exception as e:                       # BLE hiccups shouldn't kill the node
            self.get_logger().error(f"'{cmd}' failed: {e}")

    # --- expressive behaviors -------------------------------------------------
    def _greet(self):
        self.bolt.set_main_led(COLORS["green"])
        self.bolt.spin(360, 1)

    def _curious(self):
        self.bolt.set_main_led(COLORS["blue"])
        self.bolt.roll(0, 60, 1)

    def _alarm(self):
        for _ in range(2):
            self.bolt.set_main_led(COLORS["red"])
            self.bolt.spin(180, 1)


def main():
    rclpy.init()
    print("Scanning for BOLT — shake it awake...")
    toy = scanner.find_toy(toy_name=BOLT_NAME)
    if toy is None:
        print(f"{BOLT_NAME} not found. Shake it and retry.")
        rclpy.shutdown()
        return

    with SpheroEduAPI(toy) as bolt:
        bolt.set_main_led(COLORS["green"])           # connected indicator
        node = BoltNode(bolt)
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            bolt.set_main_led(COLORS["off"])
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()