#!/usr/bin/env python3
"""
obstacle_avoider.py
===================
Uses 2-D lidar to detect obstacles (static + dynamic) and generates a
local detour path around them.  The MPC tracker subscribes to the modified
path produced here.

Architecture
------------
  PathSmootherNode  ──/path──►  ObstacleAvoiderNode  ──/path_local──►  MPCTrackerNode
                                         ▲
                               /scan (LaserScan)

Key ideas
---------
* Keeps the full global reference path in memory.
* Each control cycle, converts the laser scan to world-frame obstacle points.
* Checks whether the next N path points are obstacle-free.
* If blocked, plans a 3-waypoint detour arc around the obstacle ONCE,
  commits to it, and keeps the same plan until the robot finishes the
  detour or the planned arc itself becomes obstructed.  Re-planning every
  cycle would cause wobble — we don't do that.
* If the obstacle is right in front of the robot it publishes a 1-point
  path so the MPC stops cleanly.
* Dynamic obstacles dropped onto the planned arc trigger one re-plan.
"""

import math
import threading
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, ReliabilityPolicy,
                       DurabilityPolicy, HistoryPolicy)

from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def yaw_from_quat(q) -> float:
    return math.atan2(2 * q.w * q.z, 1 - 2 * q.z * q.z)


def dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


# ─────────────────────────────────────────────────────────────────────────────
# Node
# ─────────────────────────────────────────────────────────────────────────────

class ObstacleAvoiderNode(Node):

    def __init__(self):
        super().__init__("obstacle_avoider_node")

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter("check_horizon",       3.0)
        self.declare_parameter("obstacle_clearance",  0.40)
        self.declare_parameter("detour_offset",       1.10)
        self.declare_parameter("control_hz",         10.0)
        self.declare_parameter("robot_radius",        0.25)
        self.declare_parameter("frame_id",           "odom")
        self.declare_parameter("lookahead_check_pts", 30)
        self.declare_parameter("scan_max_range",      3.0)
        self.declare_parameter("stop_dist",           0.18)
        self.declare_parameter("recovery_hold_sec",   0.8)
        # Anti-noise: a path point counts as blocked only when at least
        # this many scan points cluster within the safety threshold.
        self.declare_parameter("min_block_hits",      2)
        # Always dodge clockwise (perpendicular RIGHT of the path tangent).
        # Set to False to auto-pick a side based on which side is clearer.
        self.declare_parameter("clockwise_only",      True)
        # Extra distance the reconnect point must sit past the obstacle's
        # far edge along the path-forward direction.  Bumping this up
        # makes the detour merge back further down the path, which stops
        # the "detour twice in a row" behaviour on obstacles whose far
        # edge wasn't visible to the lidar at planning time.
        self.declare_parameter("merge_forward_margin", 1.20)
        # Multiplier on the arc's final waypoint forward distance.
        # >1 lengthens the arc so it ends further past the obstacle
        # before merging into the global path.
        self.declare_parameter("arc_forward_stretch",  1.40)
        # After a detour completes, suppress new-detour planning for
        # this many metres of forward travel.  Without this, lidar
        # echoes from the just-passed obstacle can trigger a spurious
        # "second detour" right after the first one finishes.
        self.declare_parameter("post_detour_cooldown_m", 1.50)

        self.check_horizon  = self.get_parameter("check_horizon").value
        self.obs_clearance  = self.get_parameter("obstacle_clearance").value
        self.detour_offset  = self.get_parameter("detour_offset").value
        hz                  = self.get_parameter("control_hz").value
        self.robot_radius   = self.get_parameter("robot_radius").value
        self.frame_id       = self.get_parameter("frame_id").value
        self.check_pts      = self.get_parameter("lookahead_check_pts").value
        self.scan_max_range = self.get_parameter("scan_max_range").value
        self.stop_dist      = self.get_parameter("stop_dist").value
        self.recovery_hold  = self.get_parameter("recovery_hold_sec").value
        self.min_block_hits = self.get_parameter("min_block_hits").value
        self.clockwise_only      = self.get_parameter("clockwise_only").value
        self.merge_forward_margin = self.get_parameter("merge_forward_margin").value
        self.arc_forward_stretch  = self.get_parameter("arc_forward_stretch").value
        self.post_detour_cooldown_m = self.get_parameter("post_detour_cooldown_m").value

        # ── State ────────────────────────────────────────────────────────────
        self.global_path: list[tuple[float, float]] = []
        self.x = self.y = self.yaw = 0.0
        self.scan: LaserScan | None = None
        self.lock = threading.Lock()

        self.detour_active = False
        self.detour_path: list[tuple[float, float]] = []
        self.last_clear_time = None
        # Number of points in the planned ARC portion of the detour
        self._arc_n_points = 0
        # Last published path content — used to suppress re-publishing
        # the same plan over and over (each republish triggers an MPC
        # progress reset and causes wobble).
        self._last_published: list[tuple[float, float]] = []

        # Monotonic progress along the global path.  Once advanced past
        # index N, never look at indices < N.  Forces in-order waypoint
        # tracking even when far off the path on a detour.
        # `None` means "not yet seeded" — _advance_progress will do a full
        # closest-point search the first time it runs after a path reset.
        self.progress_idx: int | None = None
        # Locked-in reconnect index for the current detour.
        self.committed_reconnect_idx = 0

        # Cooldown after a detour completes: the path index at which the
        # cooldown ENDS.  While progress_idx < cooldown_end_idx, suppress
        # new detour planning unless a real front-of-robot emergency
        # (front_dist < stop_dist) demands it.  Without this, lidar
        # echoes from the just-passed obstacle can trigger a spurious
        # "second detour" right after the first one finishes.
        self.cooldown_end_idx: int = 0

        # ── QoS ─────────────────────────────────────────────────────────────
        latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── Subscribers ──────────────────────────────────────────────────────
        self.create_subscription(Path,      "path", self._path_cb, latched)
        self.create_subscription(LaserScan, "scan", self._scan_cb, 10)
        self.create_subscription(Odometry,  "odom", self._odom_cb, 10)

        # ── Publishers ───────────────────────────────────────────────────────
        # Use latched QoS for visualisation topics too — RViz subscribes
        # whenever the user opens it, and a transient publisher would mean
        # the path is invisible until the next state change.
        self.local_pub        = self.create_publisher(Path,        "path_local",            latched)
        self.global_path_pub  = self.create_publisher(Path,        "/viz/global_path",      latched)
        self.detour_path_pub  = self.create_publisher(Path,        "/viz/detour_path",      latched)
        self.obstacle_pub     = self.create_publisher(MarkerArray, "/viz/obstacles",        10)
        self.waypoint_pub     = self.create_publisher(MarkerArray, "/viz/detour_waypoints", latched)

        # ── Timer ────────────────────────────────────────────────────────────
        self.create_timer(1.0 / hz, self._update)
        self.get_logger().info("ObstacleAvoiderNode ready.")

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _path_cb(self, msg: Path):
        with self.lock:
            self.global_path = [
                (p.pose.position.x, p.pose.position.y) for p in msg.poses
            ]
            # Reset progress to None so _advance_progress re-seeds via a
            # full closest-point search next cycle.
            self.progress_idx = None
            self.committed_reconnect_idx = 0
            self.cooldown_end_idx = 0
            self.detour_active = False
            self.detour_path = []
            self._arc_n_points = 0
            self._last_published = []  # force a republish next cycle
        self.get_logger().info(
            f"Global path received: {len(self.global_path)} points."
        )

    def _scan_cb(self, msg: LaserScan):
        with self.lock:
            self.scan = msg

    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose
        self.x   = p.position.x
        self.y   = p.position.y
        self.yaw = yaw_from_quat(p.orientation)

    # ── Visualisation ────────────────────────────────────────────────────────

    def _publish_viz(self, global_path, detour_path, obs_pts, waypoints=None):
        now = self.get_clock().now().to_msg()

        gp_msg = Path()
        gp_msg.header.frame_id = self.frame_id
        gp_msg.header.stamp = now
        for x, y in global_path:
            ps = PoseStamped()
            ps.header = gp_msg.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.orientation.w = 1.0
            gp_msg.poses.append(ps)
        self.global_path_pub.publish(gp_msg)

        dp_msg = Path()
        dp_msg.header.frame_id = self.frame_id
        dp_msg.header.stamp = now
        for x, y in detour_path:
            ps = PoseStamped()
            ps.header = dp_msg.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.orientation.w = 1.0
            dp_msg.poses.append(ps)
        self.detour_path_pub.publish(dp_msg)

        obs_markers = MarkerArray()
        clear = Marker()
        clear.header.frame_id = self.frame_id
        clear.header.stamp = now
        clear.action = Marker.DELETEALL
        obs_markers.markers.append(clear)
        for i, (ox, oy) in enumerate(obs_pts[:100]):
            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp = now
            m.ns = "obstacles"
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(ox)
            m.pose.position.y = float(oy)
            m.pose.position.z = 0.1
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.08
            m.color.r = 1.0
            m.color.a = 0.8
            obs_markers.markers.append(m)
        self.obstacle_pub.publish(obs_markers)

        if waypoints:
            wp_markers = MarkerArray()
            clear2 = Marker()
            clear2.header.frame_id = self.frame_id
            clear2.header.stamp = now
            clear2.action = Marker.DELETEALL
            wp_markers.markers.append(clear2)
            for i, (wx, wy) in enumerate(waypoints):
                m = Marker()
                m.header.frame_id = self.frame_id
                m.header.stamp = now
                m.ns = "detour_wp"
                m.id = i
                m.type = Marker.SPHERE
                m.action = Marker.ADD
                m.pose.position.x = float(wx)
                m.pose.position.y = float(wy)
                m.pose.position.z = 0.2
                m.pose.orientation.w = 1.0
                m.scale.x = m.scale.y = m.scale.z = 0.15
                # Orange to match the detour path colour (and not clash with
                # the cyan CSV waypoints)
                m.color.r = 1.0
                m.color.g = 0.55
                m.color.b = 0.0
                m.color.a = 1.0
                wp_markers.markers.append(m)

                t = Marker()
                t.header.frame_id = self.frame_id
                t.header.stamp = now
                t.ns = "detour_wp_labels"
                t.id = i + 1000
                t.type = Marker.TEXT_VIEW_FACING
                t.action = Marker.ADD
                t.pose.position.x = float(wx)
                t.pose.position.y = float(wy)
                t.pose.position.z = 0.4
                t.pose.orientation.w = 1.0
                t.scale.z = 0.2
                t.color.r = t.color.g = t.color.b = 1.0
                t.color.a = 1.0
                t.text = f"D{i+1}"   # D1/D2/D3 — distinct from CSV WP labels
                wp_markers.markers.append(t)
            self.waypoint_pub.publish(wp_markers)

    # ── Core update ──────────────────────────────────────────────────────────

    def _update(self):
        with self.lock:
            path = list(self.global_path)
            scan = self.scan

        if not path:
            self.get_logger().warn("No global path yet.", throttle_duration_sec=5.0)
            return
        if scan is None:
            self.get_logger().warn("No scan yet.", throttle_duration_sec=5.0)
            return

        start_idx = self._advance_progress(path)
        obs_pts = self._scan_to_world(scan)
        front_dist = self._closest_front_distance(scan)

        # ─────────────────────────────────────────────────────────────────
        # Branch A: currently on a committed detour.
        # Keep using the same plan until either (a) the robot reaches the
        # reconnect index, or (b) the planned arc itself becomes blocked.
        # Do NOT re-plan just because the original obstacle is still
        # visible — we already know about it.
        # ─────────────────────────────────────────────────────────────────
        if self.detour_active and self.detour_path:
            reached_merge = (
                self.committed_reconnect_idx > 0
                and start_idx >= self.committed_reconnect_idx
            )

            arc_len = max(1, self._arc_n_points)
            full_arc = self.detour_path[:arc_len]
            # Only check arc points that are still AHEAD of the robot.
            # Arc points already passed don't matter — checking them
            # would falsely re-trigger replanning when the lidar starts
            # to see the back face of the obstacle (which is naturally
            # close to those past arc points but no longer relevant).
            arc_pts = self._points_ahead_of_robot(full_arc)
            # Use a TIGHTER threshold here than the planning threshold.
            # The arc was placed at `detour_offset` from the obstacle, so
            # the original obstacle's scan points are nominally clear of
            # the arc.  Only re-plan if something is *really* close to
            # the arc (i.e. a new obstacle), not if scan noise jitters
            # the original obstacle's centroid by a few cm.
            arc_threshold = self.robot_radius + 0.5 * self.obs_clearance
            arc_blocked = self._points_within(arc_pts, obs_pts, arc_threshold)

            if front_dist < self.stop_dist:
                self.get_logger().warn(
                    f"Obstacle within {front_dist:.2f}m — emergency hold.",
                    throttle_duration_sec=1.0,
                )
                self._publish_stop()
                self._publish_viz(path, [], obs_pts, None)
                return

            if arc_blocked:
                # Something new on the planned arc → re-plan once
                self.get_logger().warn(
                    "Detour arc itself is blocked — replanning.",
                    throttle_duration_sec=1.0,
                )
                ahead = path[start_idx: start_idx + self.check_pts + 1]
                _, block_idx = self._is_path_blocked(ahead, obs_pts)
                new_detour = self._plan_detour(
                    path, start_idx, block_idx, obs_pts
                )
                if new_detour:
                    self.detour_path = new_detour
                # If replan failed, keep the old detour (better than nothing)
                self._publish_if_changed(self.detour_path)
                self._publish_viz(
                    path, self.detour_path, obs_pts,
                    self._extract_arc_waypoints(self.detour_path),
                )
                return

            if reached_merge:
                self.get_logger().info("Detour complete, resuming global path.")
                self.detour_active = False
                self.detour_path = []
                # Set cooldown: don't plan a new detour until the robot has
                # advanced cooldown_m metres past the current progress.
                # Approximate path resolution as 5 cm/point.
                cooldown_pts = max(1, int(self.post_detour_cooldown_m / 0.05))
                self.cooldown_end_idx = start_idx + cooldown_pts
                self.committed_reconnect_idx = 0
                self._arc_n_points = 0
                self.last_clear_time = None
                self._publish_if_changed(path)
                self._publish_viz(path, path[start_idx:], obs_pts, None)
                return

            # Still on the detour, arc still clear → keep publishing the
            # SAME committed plan.  This is the key change: stable plan,
            # no wobble.
            self._publish_if_changed(self.detour_path)
            self._publish_viz(
                path, self.detour_path, obs_pts,
                self._extract_arc_waypoints(self.detour_path),
            )

            self.get_logger().info(
                f"Robot=({self.x:.2f},{self.y:.2f}) "
                f"progress={start_idx} "
                f"reconnect={self.committed_reconnect_idx} "
                f"following committed detour",
                throttle_duration_sec=1.0,
            )
            return

        # ─────────────────────────────────────────────────────────────────
        # Branch B: not detouring.  Check the global path ahead.
        # ─────────────────────────────────────────────────────────────────
        ahead = path[start_idx: start_idx + self.check_pts + 1]
        blocked, block_idx = self._is_path_blocked(ahead, obs_pts)

        self.get_logger().info(
            f"Robot=({self.x:.2f},{self.y:.2f}) "
            f"progress={start_idx}/{len(path)} "
            f"front={front_dist:.2f}m blocked={blocked}",
            throttle_duration_sec=1.0,
        )

        if blocked:
            # Post-detour cooldown: suppress fresh detour planning for a
            # short window of forward travel, unless something is genuinely
            # right in front of the robot.  This kills the spurious "second
            # detour" right after completing a real one.
            in_cooldown = start_idx < self.cooldown_end_idx
            real_emergency = front_dist < self.stop_dist + 0.10
            if in_cooldown and not real_emergency:
                self.get_logger().info(
                    f"In post-detour cooldown ({start_idx}/"
                    f"{self.cooldown_end_idx}) — ignoring suspected block "
                    f"(front={front_dist:.2f}m).",
                    throttle_duration_sec=1.0,
                )
                self._publish_if_changed(path)
                self._publish_viz(path, path[start_idx:], obs_pts, None)
                return

            self.get_logger().info(
                f"Path blocked at idx={start_idx + block_idx} — planning detour."
            )
            new_detour = self._plan_detour(
                path, start_idx, block_idx, obs_pts
            )
            if not new_detour:
                self.get_logger().warn(
                    f"Detour planning failed (front={front_dist:.2f}m) — "
                    f"holding.",
                    throttle_duration_sec=1.0,
                )
                self._publish_stop()
                self._publish_viz(path, [], obs_pts, None)
                return

            self.detour_active = True
            self.detour_path = new_detour
            self._publish_if_changed(self.detour_path)
            self._publish_viz(
                path, self.detour_path, obs_pts,
                self._extract_arc_waypoints(self.detour_path),
            )
            return

        # Path ahead is clear → cooldown no longer needed (we're done with
        # the previous obstacle either way).
        if self.cooldown_end_idx > 0 and start_idx >= self.cooldown_end_idx:
            self.cooldown_end_idx = 0

        # Normal case: publish the FULL global path (unchanging content).
        # The MPC tracks progress on its own; publishing a robot-relative
        # slice every cycle would change content every tick and force the
        # MPC to reset progress, which causes the post-obstacle wobble.
        self._publish_if_changed(path)
        self._publish_viz(path, path[start_idx:], obs_pts, None)

    def _advance_progress(self, path: list[tuple[float, float]]) -> int:
        """
        Return the current monotonic progress index along `path`.  Once
        seeded, only ever moves forward.

        Initial seed: when a new path arrives, `progress_idx` is reset
        to None and we do a *full* scan of the path to find the closest
        point.  After that, we only search forward in a bounded window.
        Without this seeding, the index would be stuck at 0 forever if
        the path's index 0 happens to be far from where the robot starts.
        """
        if not path:
            return 0

        # First call after a path reset: seed to the globally-closest point
        if self.progress_idx is None:
            rob = (self.x, self.y)
            self.progress_idx = min(range(len(path)),
                                    key=lambda i: dist(rob, path[i]))
            self.get_logger().info(
                f"Path progress seeded to idx {self.progress_idx}/{len(path)} "
                f"(robot at {rob[0]:.2f},{rob[1]:.2f}, "
                f"path[{self.progress_idx}]={path[self.progress_idx]})"
            )
            return self.progress_idx

        # Clamp in case the path got shorter
        self.progress_idx = min(self.progress_idx, len(path) - 1)

        # Forward-only search in a bounded window.  Window is sized
        # generously so a fast-moving robot doesn't outrun it.
        window_end = min(len(path), self.progress_idx + 100)
        rob = (self.x, self.y)
        best_i = self.progress_idx
        best_d = dist(rob, path[best_i])
        for i in range(self.progress_idx + 1, window_end):
            d = dist(rob, path[i])
            if d < best_d:
                best_d = d
                best_i = i
        if best_i > self.progress_idx:
            self.progress_idx = best_i
        return self.progress_idx

    # ── Scan → world-frame points ────────────────────────────────────────────

    def _scan_to_world(self, scan: LaserScan) -> list[tuple[float, float]]:
        pts = []
        for i, r in enumerate(scan.ranges):
            if not math.isfinite(r) or r < scan.range_min or r > scan.range_max:
                continue
            if r > self.scan_max_range:
                continue
            ang_robot = scan.angle_min + i * scan.angle_increment
            ang_world = ang_robot + self.yaw
            wx = self.x + r * math.cos(ang_world)
            wy = self.y + r * math.sin(ang_world)
            pts.append((wx, wy))
        return pts

    def _closest_front_distance(self, scan: LaserScan,
                                cone_deg: float = 30.0) -> float:
        cone = math.radians(cone_deg)
        best = float("inf")
        for i, r in enumerate(scan.ranges):
            if not math.isfinite(r) or r < scan.range_min or r > scan.range_max:
                continue
            ang = scan.angle_min + i * scan.angle_increment
            ang = math.atan2(math.sin(ang), math.cos(ang))
            if abs(ang) <= cone and r < best:
                best = r
        return best

    # ── Path / detour helpers ────────────────────────────────────────────────

    def _find_closest_idx(self, path, pt) -> int:
        return min(range(len(path)), key=lambda i: dist(pt, path[i]))

    def _is_path_blocked(
        self,
        path_segment: list[tuple[float, float]],
        obs_pts: list[tuple[float, float]],
    ) -> tuple[bool, int]:
        """
        A path point is blocked only when at least `min_block_hits` scan
        points cluster within the safety threshold.  Single-point sparkles
        are ignored, which removes a major source of phantom blocks.
        """
        threshold = self.robot_radius + self.obs_clearance
        for i, wp in enumerate(path_segment):
            hits = 0
            for op in obs_pts:
                if dist(wp, op) < threshold:
                    hits += 1
                    if hits >= self.min_block_hits:
                        return True, i
        return False, len(path_segment)

    def _points_within(
        self,
        path_segment: list[tuple[float, float]],
        obs_pts: list[tuple[float, float]],
        threshold: float,
    ) -> bool:
        """
        Like `_is_path_blocked` but with an explicit threshold.  Used to
        check whether the planned detour arc has been newly blocked,
        with a tighter threshold than the planning one — so the original
        obstacle (which is `detour_offset` away) doesn't trip it.
        """
        for wp in path_segment:
            hits = 0
            for op in obs_pts:
                if dist(wp, op) < threshold:
                    hits += 1
                    if hits >= self.min_block_hits:
                        return True
        return False

    def _points_ahead_of_robot(
        self, path_segment: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        """
        Return only the path points that are in front of the robot,
        relative to its current heading.  A point is "ahead" if the
        vector from robot to point has a positive component along the
        robot's forward direction.
        """
        fwd_x = math.cos(self.yaw)
        fwd_y = math.sin(self.yaw)
        return [
            wp for wp in path_segment
            if (wp[0] - self.x) * fwd_x + (wp[1] - self.y) * fwd_y > 0.0
        ]

    def _extract_arc_waypoints(self, detour_path):
        if not detour_path or len(detour_path) < 3:
            return None
        n = len(detour_path)
        return [detour_path[0], detour_path[n // 2], detour_path[-1]]

    def _plan_detour(self, full_path, start_idx, block_idx, obs_pts):
        """
        Plan a 3-waypoint arc detour around the closest obstacle.

        Returns: detour path (dense_arc + global path tail) on success,
                 empty list on failure (caller must NOT enter detour state).
        """
        threshold = self.robot_radius + self.obs_clearance
        blocked_path_idx = start_idx + block_idx

        near_obs = [p for p in obs_pts if dist((self.x, self.y), p) < 2.5]
        if not near_obs:
            self.get_logger().warn("Cannot plan detour: no nearby obstacles.")
            return []

        # Cluster around the closest hit → obstacle centroid.  Use a
        # generous cluster radius so wider obstacles get treated as one
        # blob instead of as multiple little ones.
        closest_obs = min(near_obs, key=lambda p: dist((self.x, self.y), p))
        same_obs = [p for p in near_obs if dist(closest_obs, p) < 0.8]
        obs_cx = sum(p[0] for p in same_obs) / len(same_obs)
        obs_cy = sum(p[1] for p in same_obs) / len(same_obs)

        # Forward direction = path tangent at the block point
        path_fwd_x, path_fwd_y = self._path_tangent(full_path, blocked_path_idx)

        # Obstacle's *forward extent* — how far past the centroid the
        # obstacle reaches along the path-forward direction.  Without
        # this, a long obstacle's far edge sticks out past wp3 and the
        # robot triggers a second detour as it merges back.
        forward_projections = [
            (p[0] - obs_cx) * path_fwd_x + (p[1] - obs_cy) * path_fwd_y
            for p in same_obs
        ]
        obs_forward_extent = max(forward_projections) if forward_projections else 0.0
        # Also remember sideways extent (perpendicular to path) for the
        # same reason on the dodge axis.
        obs_side_extent = max(
            abs((p[0] - obs_cx) * (-path_fwd_y) +
                (p[1] - obs_cy) *   path_fwd_x)
            for p in same_obs
        ) if same_obs else 0.0

        # Clockwise = perpendicular RIGHT of forward direction
        perp_right = ( path_fwd_y, -path_fwd_x)
        perp_left  = (-path_fwd_y,  path_fwd_x)

        if self.clockwise_only:
            dodge = perp_right
            side_name = "clockwise (right)"
        else:
            rob_to_obs_x = obs_cx - self.x
            rob_to_obs_y = obs_cy - self.y
            cross = path_fwd_x * rob_to_obs_y - path_fwd_y * rob_to_obs_x
            primary_dodge   = perp_right if cross > 0 else perp_left
            primary_name    = "right"   if cross > 0 else "left"
            secondary_dodge = perp_left  if cross > 0 else perp_right
            secondary_name  = "left"    if cross > 0 else "right"
            dodge, side_name = self._pick_clear_side(
                primary_dodge, primary_name,
                secondary_dodge, secondary_name,
                obs_cx, obs_cy, obs_pts, threshold,
            )

        # Effective lateral clearance: detour_offset PLUS the obstacle's
        # half-width in the perpendicular direction.  This way the arc
        # gives the same clearance regardless of how big the object is.
        d_side = self.detour_offset + obs_side_extent
        # Effective forward clearance for wp3.  The `arc_forward_stretch`
        # multiplier (>1) extends the arc further past the obstacle,
        # giving the merge-back point room to rejoin the path AFTER the
        # robot has cleared the obstacle's body — not while it's still
        # alongside it.
        d_fwd = (self.detour_offset + obs_forward_extent) * self.arc_forward_stretch

        # 3 waypoints: beside robot, beside obstacle, past obstacle
        wp1 = (self.x + dodge[0] * d_side,
               self.y + dodge[1] * d_side)
        wp2 = (obs_cx + dodge[0] * d_side,
               obs_cy + dodge[1] * d_side)
        # wp3 sits well past the obstacle's far edge, biased back toward
        # the path so the merge-in is gentle.
        wp3 = (obs_cx + path_fwd_x * d_fwd + dodge[0] * d_side * 0.4,
               obs_cy + path_fwd_y * d_fwd + dodge[1] * d_side * 0.4)

        # Push waypoints further outward if any are too close to scan hits
        safe_waypoints = []
        for wp in (wp1, wp2, wp3):
            pushed = wp
            for extra in (0.0, 0.15, 0.30, 0.50, 0.70, 1.00, 1.30):
                cand = (wp[0] + dodge[0] * extra, wp[1] + dodge[1] * extra)
                if all(dist(cand, op) > threshold for op in obs_pts):
                    pushed = cand
                    break
            safe_waypoints.append(pushed)

        # Reconnect index: must be (a) clear of scan hits, AND (b) past
        # *wp3* in the path-forward direction (not just past the obstacle).
        # The previous logic could pick a reconnect point BEHIND wp3,
        # which made the dense_arc end at wp3 and then jump backwards to
        # the reconnect — producing the zigzag in image 1.
        wp3_x, wp3_y = safe_waypoints[2]
        wp3_past = ((wp3_x - obs_cx) * path_fwd_x +
                    (wp3_y - obs_cy) * path_fwd_y)
        # Require reconnect at least 0.30 m past wp3, AND past the
        # obstacle by `merge_forward_margin`, whichever is further.
        required_past = max(
            wp3_past + 0.30,
            obs_forward_extent + self.merge_forward_margin,
        )
        search_start = max(blocked_path_idx, self.committed_reconnect_idx)
        reconnect_idx = search_start
        while reconnect_idx < len(full_path):
            wp = full_path[reconnect_idx]
            past = ((wp[0] - obs_cx) * path_fwd_x +
                    (wp[1] - obs_cy) * path_fwd_y)
            clear = all(dist(wp, op) > threshold for op in obs_pts)
            if past > required_past and clear:
                break
            reconnect_idx += 1
        if reconnect_idx >= len(full_path):
            reconnect_idx = len(full_path) - 1

        # Commit it
        self.committed_reconnect_idx = max(self.committed_reconnect_idx,
                                           reconnect_idx)

        # Densify the arc, AND add a smooth merge from wp3 to the
        # reconnect point so the path doesn't have a sharp corner where
        # the arc ends and the global path begins.
        robot_pt = (self.x, self.y)
        arc_pts = [robot_pt] + safe_waypoints
        dense_arc = []
        for i in range(len(arc_pts) - 1):
            dense_arc += self._interpolate(arc_pts[i], arc_pts[i + 1], 0.05)
        dense_arc.append(arc_pts[-1])
        # Smooth merge wp3 → path[reconnect_idx]
        merge_segment = self._interpolate(
            arc_pts[-1], full_path[reconnect_idx], 0.05
        )
        dense_arc += merge_segment

        # Track arc length so we can later check just the arc for blocks
        self._arc_n_points = len(dense_arc)

        detour = dense_arc + full_path[reconnect_idx:]

        self.get_logger().info(
            f"Detour planned: side={side_name} "
            f"obs=({obs_cx:.2f},{obs_cy:.2f}) "
            f"obs_extent fwd={obs_forward_extent:.2f}m side={obs_side_extent:.2f}m "
            f"d_side={d_side:.2f}m d_fwd={d_fwd:.2f}m "
            f"arc_pts={len(dense_arc)} "
            f"reconnect={reconnect_idx}/{len(full_path)} "
            f"(req_past={required_past:.2f}m)",
        )
        return detour

    def _path_tangent(self, path, idx):
        if len(path) < 2:
            return (math.cos(self.yaw), math.sin(self.yaw))
        i0 = max(0, min(idx,     len(path) - 2))
        i1 = max(1, min(idx + 1, len(path) - 1))
        dx = path[i1][0] - path[i0][0]
        dy = path[i1][1] - path[i0][1]
        n  = math.hypot(dx, dy)
        if n < 1e-6:
            return (math.cos(self.yaw), math.sin(self.yaw))
        return (dx / n, dy / n)

    def _pick_clear_side(self, primary, primary_name,
                         secondary, secondary_name,
                         obs_cx, obs_cy, obs_pts, threshold):
        for dodge, name in ((primary, primary_name), (secondary, secondary_name)):
            wp2 = (obs_cx + dodge[0] * self.detour_offset,
                   obs_cy + dodge[1] * self.detour_offset)
            if all(dist(wp2, op) > threshold for op in obs_pts):
                return dodge, name
        return primary, primary_name

    @staticmethod
    def _interpolate(p1, p2, resolution=0.05):
        d = dist(p1, p2)
        if d < resolution:
            return [p1]
        n = max(2, int(d / resolution))
        return [
            (p1[0] + (p2[0] - p1[0]) * t / n,
             p1[1] + (p2[1] - p1[1]) * t / n)
            for t in range(n)
        ]

    # ── Output ───────────────────────────────────────────────────────────────

    def _publish_if_changed(self, path: list[tuple[float, float]]):
        """
        Publish to /path_local only if the path content has changed.
        Each publish triggers an MPC progress reset, so suppressing
        identical re-publishes prevents the MPC from constantly
        re-evaluating from scratch (which causes wobble).
        """
        if not path:
            return
        if self._paths_equal(path, self._last_published):
            return  # latched QoS keeps delivering the previous one

        msg = Path()
        msg.header.frame_id = self.frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        for x, y in path:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.local_pub.publish(msg)
        self._last_published = list(path)

    @staticmethod
    def _paths_equal(a, b, tol=1e-3):
        if a is None or b is None:
            return False
        if len(a) != len(b):
            return False
        for (ax, ay), (bx, by) in zip(a, b):
            if abs(ax - bx) > tol or abs(ay - by) > tol:
                return False
        return True

    def _publish_stop(self):
        """1-point path at robot position so MPC sees zero distance."""
        msg = Path()
        msg.header.frame_id = self.frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        ps = PoseStamped()
        ps.header = msg.header
        ps.pose.position.x = float(self.x)
        ps.pose.position.y = float(self.y)
        ps.pose.orientation.w = 1.0
        msg.poses.append(ps)
        self.local_pub.publish(msg)
        # Update the cache so next non-stop publish always goes through
        self._last_published = [(self.x, self.y)]


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleAvoiderNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
