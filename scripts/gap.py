import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped


MAX_SPEED       = 20000.0    # m/s — top speed on clear straight
MIN_SPEED       = 1.5       # m/s — minimum through tight corners
SAFE_DIST       = 0.5       # m  — bubble radius to zero out around closest point
MAX_STEER       = 0.1       # rad — physical steering limit
STEER_GAIN      = 0.6       # how aggressively to steer toward gap center
SPEED_GAIN      = 0.4       # how much gap depth boosts speed
FRONT_CONE      = 60        # degrees each side of forward for speed scaling


class GapFollower(Node):
    def __init__(self):
        super().__init__('gap_follower')
        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)

    def scan_cb(self, msg):
        ranges = np.array(msg.ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))

        ranges = np.clip(ranges, 0.0, msg.range_max)
        ranges[~np.isfinite(ranges)] = 0.0

        # ── safety bubble: zero out SAFE_DIST around closest ─
        closest = np.argmin(ranges)
        bubble_angle = np.arctan2(SAFE_DIST, max(ranges[closest], 0.01))
        angle_res = (msg.angle_max - msg.angle_min) / len(ranges)
        bubble_px = int(bubble_angle / angle_res)
        lo = max(0, closest - bubble_px)
        hi = min(len(ranges) - 1, closest + bubble_px)
        ranges[lo:hi+1] = 0.0

        # ── find largest gap (contiguous non-zero block) ─────
        gaps = []
        in_gap = False
        g_start = 0
        for i, r in enumerate(ranges):
            if r > 0 and not in_gap:
                g_start = i
                in_gap = True
            elif r == 0 and in_gap:
                gaps.append((g_start, i - 1))
                in_gap = False
        if in_gap:
            gaps.append((g_start, len(ranges) - 1))

        if not gaps:
            self._publish(0.0, MIN_SPEED * 0.5)
            return

        best       = max(gaps, key=lambda g: (g[1] - g[0]))
        gap_center = (best[0] + best[1]) // 2
        gap_angle  = angles[gap_center]

        # ── steer toward gap center ──────────────────────────
        steer = np.clip(STEER_GAIN * gap_angle, -MAX_STEER, MAX_STEER)

        # ── speed: based on depth ahead + how much we're turning
        front_mask  = np.abs(angles) < np.deg2rad(FRONT_CONE)
        front_depth = np.mean(ranges[front_mask & (ranges > 0)]) if np.any(front_mask & (ranges > 0)) else 1.0
        speed = MIN_SPEED + SPEED_GAIN * front_depth
        speed = np.clip(speed, MIN_SPEED, MAX_SPEED)
        speed *= max(0.3, 1.0 - abs(steer) / MAX_STEER * 0.65)

        self._publish(float(steer), float(speed))

    def _publish(self, steer, speed):
        msg = AckermannDriveStamped()
        msg.drive.steering_angle = steer
        msg.drive.speed = speed
        self.pub.publish(msg)


def main():
    rclpy.init()
    rclpy.spin(GapFollower())
    rclpy.shutdown()

if __name__ == '__main__':
    main()