#!/usr/bin/env python3
"""
path_smoother.py
================
ROS 2 node that reads waypoints from a CSV, generates a smooth path using
cubic B-spline interpolation, and publishes it for the obstacle avoider /
MPC tracker.

Replaces: smoothing.cpp (DynamicPathPlanner + PathSmoother)
"""

import csv
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, ReliabilityPolicy,
                       DurabilityPolicy, HistoryPolicy)

import numpy as np
from scipy.interpolate import splprep, splev

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(filepath: str) -> list[tuple[float, float]]:
    """Read x,y pairs from a CSV file (one per line, comma-separated)."""
    waypoints = []
    with open(filepath, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                try:
                    waypoints.append((float(row[0]), float(row[1])))
                except ValueError:
                    pass  # skip header / bad lines
    return waypoints


def smooth_path(
    waypoints: list[tuple[float, float]],
    resolution: float = 0.05,
    smoothing_factor: float | None = None,
    spline_degree: int = 3,
) -> list[tuple[float, float]]:
    """
    Fit a cubic B-spline through the waypoints and re-sample at `resolution`
    metre intervals along arc length.
    """
    if len(waypoints) < 2:
        return waypoints

    xs = np.array([p[0] for p in waypoints])
    ys = np.array([p[1] for p in waypoints])

    # Remove duplicate consecutive points (splprep requirement)
    diffs = np.hypot(np.diff(xs), np.diff(ys))
    keep = np.concatenate(([True], diffs > 1e-6))
    xs, ys = xs[keep], ys[keep]

    if len(xs) < 2:
        return waypoints

    # Clamp degree to number of points
    k = min(spline_degree, len(xs) - 1)
    s = smoothing_factor if smoothing_factor is not None else 0.0

    tck, _ = splprep([xs, ys], s=s, k=k, per=False)

    # Estimate arc length at high density, then re-sample evenly
    u_dense = np.linspace(0, 1, max(len(waypoints) * 50, 500))
    x_dense, y_dense = splev(u_dense, tck)
    arc = np.concatenate(([0], np.cumsum(np.hypot(np.diff(x_dense),
                                                  np.diff(y_dense)))))
    total_length = arc[-1]

    if total_length < 1e-6:
        return waypoints

    n_pts = max(2, int(math.ceil(total_length / resolution)))
    u_even = np.interp(np.linspace(0, total_length, n_pts), arc, u_dense)
    x_out, y_out = splev(u_even, tck)

    return list(zip(x_out.tolist(), y_out.tolist()))


def path_to_msg(
    points: list[tuple[float, float]],
    frame_id: str,
    stamp,
) -> Path:
    """Convert (x, y) list to nav_msgs/Path."""
    msg = Path()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    for x, y in points:
        ps = PoseStamped()
        ps.header = msg.header
        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)
        ps.pose.orientation.w = 1.0
        msg.poses.append(ps)
    return msg


def waypoints_to_markers(
    points: list[tuple[float, float]],
    frame_id: str,
    stamp,
) -> MarkerArray:
    """
    Build a MarkerArray with one numbered sphere + text label per waypoint.
    Used to visualise the original CSV waypoints in RViz.
    """
    markers = MarkerArray()

    # Clear any previously-published markers in this namespace
    clear = Marker()
    clear.header.frame_id = frame_id
    clear.header.stamp = stamp
    clear.action = Marker.DELETEALL
    markers.markers.append(clear)

    for i, (x, y) in enumerate(points):
        # Sphere at the waypoint
        sphere = Marker()
        sphere.header.frame_id = frame_id
        sphere.header.stamp = stamp
        sphere.ns = "csv_waypoints"
        sphere.id = i
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD
        sphere.pose.position.x = float(x)
        sphere.pose.position.y = float(y)
        sphere.pose.position.z = 0.15
        sphere.pose.orientation.w = 1.0
        sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.20
        # Cyan-ish so they pop against the green smoothed path
        sphere.color.r = 0.20
        sphere.color.g = 0.85
        sphere.color.b = 1.00
        sphere.color.a = 1.0
        markers.markers.append(sphere)

        # Text label "WP1", "WP2", ...
        text = Marker()
        text.header.frame_id = frame_id
        text.header.stamp = stamp
        text.ns = "csv_waypoint_labels"
        text.id = i
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = float(x)
        text.pose.position.y = float(y)
        text.pose.position.z = 0.45
        text.pose.orientation.w = 1.0
        text.scale.z = 0.25  # text height
        text.color.r = 1.0
        text.color.g = 1.0
        text.color.b = 1.0
        text.color.a = 1.0
        text.text = f"WP{i + 1}"
        markers.markers.append(text)

    return markers


# ─────────────────────────────────────────────────────────────────────────────
# ROS 2 Node
# ─────────────────────────────────────────────────────────────────────────────

class PathSmootherNode(Node):
    """
    Loads waypoints from CSV, smooths them with a B-spline, and publishes:
      /waypoints  – raw waypoints (for RViz visualisation)
      /path       – dense smoothed path (for obstacle_avoider), latched QoS
    """

    def __init__(self):
        super().__init__("path_smoother_node")

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter("path_file", "waypoints/waypoint.csv")
        self.declare_parameter("frame_id", "odom")
        self.declare_parameter("path_resolution", 0.05)
        self.declare_parameter("smoothing_factor", 0.0)
        self.declare_parameter("spline_degree", 3)
        self.declare_parameter("publish_delay_sec", 2.0)

        path_file     = self.get_parameter("path_file").value
        frame_id      = self.get_parameter("frame_id").value
        resolution    = self.get_parameter("path_resolution").value
        smooth_factor = self.get_parameter("smoothing_factor").value
        spline_deg    = self.get_parameter("spline_degree").value
        delay_sec     = self.get_parameter("publish_delay_sec").value

        self.frame_id = frame_id

        # ── Publishers ───────────────────────────────────────────────────────
        latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.raw_pub      = self.create_publisher(Path,        "/waypoints",          latched_qos)
        self.path_pub     = self.create_publisher(Path,        "path",                latched_qos)
        self.markers_pub  = self.create_publisher(MarkerArray, "/viz/csv_waypoints",  latched_qos)

        # ── Load waypoints ───────────────────────────────────────────────────
        try:
            waypoints = load_csv(path_file)
        except FileNotFoundError:
            self.get_logger().error(f"Cannot open waypoint file: {path_file}")
            return

        if not waypoints:
            self.get_logger().error("No valid waypoints found in CSV.")
            return

        self.get_logger().info(
            f"Loaded {len(waypoints)} raw waypoints from {path_file}"
        )

        # ── Smooth path ──────────────────────────────────────────────────────
        smooth = smooth_path(waypoints, resolution, smooth_factor, spline_deg)
        self.get_logger().info(
            f"Smoothed path: {len(smooth)} points at {resolution} m resolution"
        )

        # Cache the messages so we can republish at any time
        self._raw_pts    = waypoints
        self._smooth_pts = smooth

        # Publish once immediately (latched QoS will hold it for late subs)
        self._publish_paths()

        # Republish once after a short delay so any subscriber that came up
        # late (e.g. RViz, MPC node) is guaranteed to get it.
        self._delay_timer = self.create_timer(delay_sec, self._delayed_publish)
        self._published_delayed = False

    def _publish_paths(self):
        now = self.get_clock().now().to_msg()
        self.raw_pub.publish(path_to_msg(self._raw_pts,    self.frame_id, now))
        self.path_pub.publish(path_to_msg(self._smooth_pts, self.frame_id, now))
        self.markers_pub.publish(
            waypoints_to_markers(self._raw_pts, self.frame_id, now)
        )

    def _delayed_publish(self):
        if self._published_delayed:
            return
        self._publish_paths()
        self.get_logger().info("Path republished. Robot will begin tracking.")
        self._published_delayed = True
        # Stop the timer so we don't keep republishing forever
        self._delay_timer.cancel()


def main(args=None):
    rclpy.init(args=args)
    node = PathSmootherNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
