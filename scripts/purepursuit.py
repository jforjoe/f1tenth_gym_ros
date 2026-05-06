#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import csv
import os

from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
from tf_transformations import euler_from_quaternion
from ament_index_python.packages import get_package_share_directory


    # config = 

WAYPOINTS_CSV   = os.path.join(
        get_package_share_directory('f1tenth_gym_ros'),
        'config',
        'waypoints.csv'
        )
LOOKAHEAD       = 1.5
WHEELBASE       = 0.33
MAX_SPEED       = 8.0
MIN_SPEED       = 1.5
BRAKE_DISTANCE  = 1.2
BRAKE_SPEED     = 1.0
STEER_GAIN      = 1.0
MAX_STEER       = 0.4


class PurePursuit(Node):
    def __init__(self):
        super().__init__('pure_pursuit')

        self.waypoints = self._load_waypoints()
        self.pos = np.array([0.0, 0.0])
        self.yaw = 0.0
        self.obstacle_ahead = False

        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.create_timer(0.05, self.control_loop)

    def _load_waypoints(self):
        pts = []
        with open(WAYPOINTS_CSV) as f:
            for row in csv.DictReader(f):
                pts.append([float(row['x_m']), float(row['y_m']), float(row['velocity_mps'])])
        return np.array(pts)

    def odom_cb(self, msg):
        self.pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        q = msg.pose.pose.orientation
        _, _, self.yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

    def scan_cb(self, msg):
        ranges = np.array(msg.ranges)
        front  = np.concatenate([ranges[-30:], ranges[:30]])
        valid  = front[np.isfinite(front)]
        self.obstacle_ahead = len(valid) > 0 and valid.min() < BRAKE_DISTANCE

    def _get_lookahead(self):
        dists = np.linalg.norm(self.waypoints[:, :2] - self.pos, axis=1)
        closest = np.argmin(dists)
        n = len(self.waypoints)
        for i in range(n):
            idx = (closest + i) % n
            if np.linalg.norm(self.waypoints[idx, :2] - self.pos) >= LOOKAHEAD:
                return self.waypoints[idx]
        return self.waypoints[closest]

    def control_loop(self):
        target = self._get_lookahead()
        tx, ty, target_v = target

        dx = tx - self.pos[0]
        dy = ty - self.pos[1]
        local_x =  np.cos(self.yaw) * dx + np.sin(self.yaw) * dy
        local_y = -np.sin(self.yaw) * dx + np.cos(self.yaw) * dy

        curvature = 2.0 * local_y / (local_x**2 + local_y**2 + 1e-6)
        steer = np.clip(STEER_GAIN * WHEELBASE * curvature, -MAX_STEER, MAX_STEER)

        speed = BRAKE_SPEED if self.obstacle_ahead else np.clip(target_v, MIN_SPEED, MAX_SPEED)
        speed *= max(0.3, 1.0 - abs(steer) / MAX_STEER * 0.5)

        msg = AckermannDriveStamped()
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steer)
        self.drive_pub.publish(msg)


def main():
    rclpy.init()
    rclpy.spin(PurePursuit())
    rclpy.shutdown()

if __name__ == '__main__':
    main()