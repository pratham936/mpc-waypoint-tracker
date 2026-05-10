#!/usr/bin/env python3
"""
mpc_tracker.py
==============
ROS 2 MPC trajectory tracker – replaces pure_pursuit.cpp.

Architecture
------------
  /path_local  (nav_msgs/Path, from ObstacleAvoiderNode)
       │
       ▼
  MPCTrackerNode   ──  cmd_vel  (geometry_msgs/Twist)
       ▲
  /odom  (nav_msgs/Odometry)
  /scan  (sensor_msgs/LaserScan)   ← emergency hard-stop only

MPC formulation  (unicycle model)
----------------------------------
  State   : [x, y, θ]
  Control : [v, ω]   (linear / angular velocity)

  Kinematic model (Euler, dt):
      x_{k+1}  = x_k  + v_k * cos(θ_k) * dt
      y_{k+1}  = y_k  + v_k * sin(θ_k) * dt
      θ_{k+1}  = θ_k  + ω_k * dt

  Cost:
      J = Σ_{k=0}^{N-1}  Q  * ||pos_k - ref_k||²
                       + Q_yaw * (θ_k - ref_θ_k)²
                       + R_v   * v_k²
                       + R_w   * ω_k²
                       + R_Δv  * (v_k - v_{k-1})²
                       + R_Δw  * (ω_k - ω_{k-1})²
        + Q_N * ||pos_N - ref_N||²

  Constraints:
      v  ∈ [-v_max,  v_max]
      ω  ∈ [-ω_max,  ω_max]
      Δv ∈ [-dv_max, dv_max]
      Δω ∈ [-dω_max, dω_max]

Solver: CasADi + IPOPT (interior-point, open-source).
"""

import math
import threading

import numpy as np
import casadi as ca
import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, ReliabilityPolicy,
                       DurabilityPolicy, HistoryPolicy)

from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def yaw_from_quat(q) -> float:
    return math.atan2(2 * q.w * q.z, 1 - 2 * q.z * q.z)


def dist2d(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


# ─────────────────────────────────────────────────────────────────────────────
# MPC Solver (built once, called every cycle)
# ─────────────────────────────────────────────────────────────────────────────

class MPCSolver:
    """
    Builds a CasADi IPOPT NLP for the unicycle MPC problem and exposes
    a `solve(state, ref_traj, prev_v, prev_w)` method.
    """

    def __init__(
        self,
        N: int   = 10,       # horizon steps
        dt: float = 0.1,     # seconds per step
        v_max: float  = 0.5,
        w_max: float  = 0.8,
        dv_max: float = 0.15,
        dw_max: float = 0.30,
        # Cost weights
        q_pos: float  = 5.0,
        q_yaw: float  = 1.0,
        r_v: float    = 0.1,
        r_w: float    = 0.2,
        r_dv: float   = 0.5,
        r_dw: float   = 0.5,
        q_terminal: float = 10.0,
    ):
        self.N  = N
        self.dt = dt
        self.v_max  = v_max
        self.w_max  = w_max
        self.dv_max = dv_max
        self.dw_max = dw_max

        # ── Decision variables ────────────────────────────────────────────────
        # Controls: [v_0, w_0, v_1, w_1, …, v_{N-1}, w_{N-1}]  → 2N vars
        U = ca.MX.sym("U", 2 * N)

        # ── Parameters ───────────────────────────────────────────────────────
        # [x0, y0, th0,  ref_x0, ref_y0, ref_th0, …, ref_xN, ref_yN, ref_thN,
        #  prev_v, prev_w]
        P = ca.MX.sym("P", 3 + 3 * (N + 1) + 2)

        x0   = P[0]; y0 = P[1]; th0 = P[2]
        prev_v = P[-2]; prev_w = P[-1]

        # ── Forward simulation ───────────────────────────────────────────────
        states = [[x0, y0, th0]]
        for k in range(N):
            vk  = U[2 * k]
            wk  = U[2 * k + 1]
            xk, yk, thk = states[-1]
            xn  = xk  + vk * ca.cos(thk) * dt
            yn  = yk  + vk * ca.sin(thk) * dt
            thn = thk + wk * dt
            states.append([xn, yn, thn])

        # ── Cost ─────────────────────────────────────────────────────────────
        cost = 0.0
        for k in range(N):
            rx  = P[3 + 3 * k]
            ry  = P[3 + 3 * k + 1]
            rth = P[3 + 3 * k + 2]
            xk, yk, thk = states[k]
            vk = U[2 * k]; wk = U[2 * k + 1]

            cost += q_pos * ((xk - rx)**2 + (yk - ry)**2)
            # Use sin/cos diff rather than raw subtraction so heading wrap
            # at ±π doesn't blow up the cost.
            cost += q_yaw * ((ca.cos(thk) - ca.cos(rth))**2 +
                             (ca.sin(thk) - ca.sin(rth))**2)
            cost += r_v * vk**2
            cost += r_w * wk**2

            if k == 0:
                cost += r_dv * (vk - prev_v)**2
                cost += r_dw * (wk - prev_w)**2
            else:
                cost += r_dv * (vk - U[2*(k-1)])**2
                cost += r_dw * (wk - U[2*(k-1)+1])**2

        # Terminal cost
        rx  = P[3 + 3 * N]
        ry  = P[3 + 3 * N + 1]
        rth = P[3 + 3 * N + 2]
        xN, yN, thN = states[N]
        cost += q_terminal * ((xN - rx)**2 + (yN - ry)**2)
        cost += q_yaw * ((ca.cos(thN) - ca.cos(rth))**2 +
                         (ca.sin(thN) - ca.sin(rth))**2)

        # ── Constraints (Δv, Δω rate limits) ──────────────────────────────────
        g, lbg, ubg = [], [], []
        for k in range(N):
            vk = U[2 * k]; wk = U[2 * k + 1]
            if k == 0:
                g  += [vk - prev_v, wk - prev_w]
            else:
                g  += [vk - U[2*(k-1)], wk - U[2*(k-1)+1]]
            lbg += [-dv_max, -dw_max]
            ubg += [ dv_max,  dw_max]

        g_cas = ca.vertcat(*g) if g else ca.MX()

        # Box constraints on controls
        lbu = np.tile([-v_max, -w_max], N)
        ubu = np.tile([ v_max,  w_max], N)

        # ── NLP ──────────────────────────────────────────────────────────────
        nlp = {"x": U, "f": cost, "p": P, "g": g_cas}
        opts = {
            "ipopt.print_level":           0,
            "ipopt.max_iter":              50,
            "ipopt.tol":                   1e-4,
            "ipopt.warm_start_init_point": "yes",
            "print_time":                  False,
        }
        self.solver = ca.nlpsol("mpc", "ipopt", nlp, opts)
        self.lbu = lbu
        self.ubu = ubu
        self.lbg = np.array(lbg, dtype=float)
        self.ubg = np.array(ubg, dtype=float)
        self.n_params = 3 + 3 * (N + 1) + 2
        self.U0 = np.zeros(2 * N)   # warm-start

    # ── Public API ────────────────────────────────────────────────────────────

    def solve(
        self,
        state: tuple[float, float, float],
        ref_traj: list[tuple[float, float, float]],
        prev_v: float,
        prev_w: float,
    ) -> tuple[float, float, list[tuple[float, float, float]]]:
        """
        Run one MPC solve step.

        Returns: (v_cmd, w_cmd, predicted_states).  predicted_states is a
        list of (x, y, theta) tuples of length N+1, starting at the
        current state.  Falls back to (0, 0, [current_state]) on failure.
        """
        assert len(ref_traj) == self.N + 1, \
            f"ref_traj must have N+1={self.N+1} rows, got {len(ref_traj)}"

        p = np.empty(self.n_params)
        p[0], p[1], p[2] = state
        for i, (rx, ry, rth) in enumerate(ref_traj):
            p[3 + 3*i]     = rx
            p[3 + 3*i + 1] = ry
            p[3 + 3*i + 2] = rth
        p[-2] = prev_v
        p[-1] = prev_w

        try:
            sol = self.solver(
                x0=self.U0,
                lbx=self.lbu,
                ubx=self.ubu,
                lbg=self.lbg,
                ubg=self.ubg,
                p=p,
            )
            U_opt = np.array(sol["x"]).flatten()
            # Warm-start: shift controls by one step, hold last value
            self.U0 = np.concatenate([U_opt[2:], U_opt[-2:]])

            # Roll out the predicted trajectory using the same kinematic
            # model as the solver, for visualisation.
            predicted = [state]
            x, y, th = state
            for k in range(self.N):
                v = float(U_opt[2 * k])
                w = float(U_opt[2 * k + 1])
                x  = x  + v * math.cos(th) * self.dt
                y  = y  + v * math.sin(th) * self.dt
                th = th + w * self.dt
                predicted.append((x, y, th))

            return float(U_opt[0]), float(U_opt[1]), predicted
        except Exception:
            self.U0 = np.zeros(2 * self.N)
            return 0.0, 0.0, [state]


# ─────────────────────────────────────────────────────────────────────────────
# ROS 2 Node
# ─────────────────────────────────────────────────────────────────────────────

class MPCTrackerNode(Node):
    """
    MPC trajectory tracker.  Reads /path_local (from ObstacleAvoiderNode),
    /odom for state, and publishes /cmd_vel.
    """

    def __init__(self):
        super().__init__("mpc_tracker_node")

        # ── Parameters ───────────────────────────────────────────────────────
        self.declare_parameter("control_hz",            10.0)
        self.declare_parameter("mpc_horizon",           10)
        self.declare_parameter("mpc_dt",                0.1)
        self.declare_parameter("v_max",                 0.5)
        self.declare_parameter("w_max",                 0.8)
        self.declare_parameter("dv_max",                0.15)
        self.declare_parameter("dw_max",                0.30)
        self.declare_parameter("goal_tolerance",        0.20)
        self.declare_parameter("q_pos",                 5.0)
        self.declare_parameter("q_yaw",                 1.0)
        self.declare_parameter("r_v",                   0.1)
        self.declare_parameter("r_w",                   0.2)
        self.declare_parameter("r_dv",                  0.5)
        self.declare_parameter("r_dw",                  0.5)
        self.declare_parameter("q_terminal",           10.0)
        self.declare_parameter("emergency_stop_dist",   0.18)
        self.declare_parameter("emergency_stop_angle",  20.0)
        self.declare_parameter("debug",                 True)

        hz            = self.get_parameter("control_hz").value
        N             = self.get_parameter("mpc_horizon").value
        dt            = self.get_parameter("mpc_dt").value
        self.v_max    = self.get_parameter("v_max").value
        self.w_max    = self.get_parameter("w_max").value
        dv            = self.get_parameter("dv_max").value
        dw            = self.get_parameter("dw_max").value
        self.goal_tol = self.get_parameter("goal_tolerance").value
        self.emg_dist = self.get_parameter("emergency_stop_dist").value
        self.emg_ang  = math.radians(self.get_parameter("emergency_stop_angle").value)
        self.debug    = self.get_parameter("debug").value

        # ── Build MPC solver ─────────────────────────────────────────────────
        self.get_logger().info("Building MPC solver (IPOPT) …")
        self.mpc = MPCSolver(
            N=N, dt=dt,
            v_max=self.v_max, w_max=self.w_max,
            dv_max=dv, dw_max=dw,
            q_pos       = self.get_parameter("q_pos").value,
            q_yaw       = self.get_parameter("q_yaw").value,
            r_v         = self.get_parameter("r_v").value,
            r_w         = self.get_parameter("r_w").value,
            r_dv        = self.get_parameter("r_dv").value,
            r_dw        = self.get_parameter("r_dw").value,
            q_terminal  = self.get_parameter("q_terminal").value,
        )
        self.N  = N
        self.dt = dt
        self.get_logger().info("MPC solver ready.")

        # ── State ─────────────────────────────────────────────────────────────
        self.path: list[tuple[float, float]] = []
        self.x = self.y = self.yaw = 0.0
        self.prev_v = self.prev_w = 0.0
        self.reached_goal = False
        self.have_odom = False
        self.lock = threading.Lock()
        self.scan: LaserScan | None = None

        # The original (unsliced) global path length, for goal detection
        self.global_path_len = 0
        self.last_path_point: tuple[float, float] | None = None

        # Monotonic progress along the published path.  Used so the MPC
        # always picks a reference point AHEAD of where it last looked,
        # even when the path is updated mid-cycle by the obstacle avoider.
        self.path_progress_idx = 0
        # Cache of the last received path so we can detect resets cleanly.
        self._path_id = 0

        # Startup gating: don't move until Gazebo has clearly come up and
        # the sim clock has been ticking for a bit.  This stops the robot
        # from twitching at (0,0,0) while Gazebo is still loading.
        self.declare_parameter("startup_wait_sec", 5.0)
        self.declare_parameter("min_odom_msgs",    20)
        self.startup_wait_sec = self.get_parameter("startup_wait_sec").value
        self.min_odom_msgs    = self.get_parameter("min_odom_msgs").value
        self.odom_count = 0
        self.first_odom_time: float | None = None
        self.startup_done = False

        # ── QoS ──────────────────────────────────────────────────────────────
        latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── Subscribers ──────────────────────────────────────────────────────
        self.create_subscription(Path,      "path_local", self._path_cb, latched)
        self.create_subscription(Odometry,  "odom",       self._odom_cb, 10)
        self.create_subscription(LaserScan, "scan",       self._scan_cb, 10)

        # ── Publisher ────────────────────────────────────────────────────────
        self.cmd_pub = self.create_publisher(Twist, "cmd_vel", 10)
        # Visualisation: predicted trajectory (orange Path) and the current
        # MPC reference target (cyan sphere).  Useful for tuning and for
        # confirming visually that the MPC is aiming at the right point.
        self.predicted_pub = self.create_publisher(Path,   "/viz/mpc_predicted", 10)
        self.target_pub    = self.create_publisher(Marker, "/viz/mpc_target",    10)

        # ── Control timer ────────────────────────────────────────────────────
        self.create_timer(1.0 / hz, self._control_step)
        self.get_logger().info("MPCTrackerNode ready.")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _path_cb(self, msg: Path):
        new_path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        with self.lock:
            old_path = self.path
            self.path = new_path
            # Only reset progress when the path SHAPE actually changes.
            if not self._paths_close(old_path, new_path):
                if new_path:
                    rob = (self.x, self.y)
                    best_i = min(range(len(new_path)),
                                 key=lambda i: math.hypot(
                                     rob[0] - new_path[i][0],
                                     rob[1] - new_path[i][1]))
                    self.path_progress_idx = best_i
                else:
                    self.path_progress_idx = 0
                self._path_id += 1
            # Track the goal (final point of the published path).  Only
            # update from REAL paths — a single-point "stop" message from
            # the obstacle_avoider is a hold-position signal, not a new
            # goal location.  Updating last_path_point from it would
            # cause a false "goal reached" trigger.
            if len(new_path) >= 2:
                self.last_path_point = new_path[-1]

    @staticmethod
    def _paths_close(a: list[tuple[float, float]],
                     b: list[tuple[float, float]],
                     tol: float = 0.02) -> bool:
        """True if two paths are essentially the same (within tol metres)."""
        if not a or not b:
            return False
        if abs(len(a) - len(b)) > 0:
            return False
        for (ax, ay), (bx, by) in zip(a, b):
            if abs(ax - bx) > tol or abs(ay - by) > tol:
                return False
        return True

    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose
        self.x   = p.position.x
        self.y   = p.position.y
        self.yaw = yaw_from_quat(p.orientation)
        self.have_odom = True
        # Startup gating: count odom messages and remember the first one.
        self.odom_count += 1
        if self.first_odom_time is None:
            self.first_odom_time = self.get_clock().now().nanoseconds * 1e-9

    def _scan_cb(self, msg: LaserScan):
        with self.lock:
            self.scan = msg

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _emergency_stop(self) -> bool:
        """
        Final-resort hard stop if something is right in front of the robot.
        Obstacle avoidance is handled upstream; this is just a safety net.
        """
        with self.lock:
            scan = self.scan
        if scan is None:
            return False
        for i, r in enumerate(scan.ranges):
            if not math.isfinite(r) or r < scan.range_min:
                continue
            ang = scan.angle_min + i * scan.angle_increment
            ang = math.atan2(math.sin(ang), math.cos(ang))
            if abs(ang) <= self.emg_ang and r < self.emg_dist:
                return True
        return False

    def _find_closest_idx(self, path: list) -> int:
        return min(range(len(path)),
                   key=lambda i: dist2d((self.x, self.y), path[i]))

    def _estimate_heading(
        self, path: list[tuple[float, float]], idx: int
    ) -> float:
        """Approximate reference heading from path tangent."""
        if idx + 1 < len(path):
            dx = path[idx + 1][0] - path[idx][0]
            dy = path[idx + 1][1] - path[idx][1]
            return math.atan2(dy, dx)
        if idx > 0:
            dx = path[idx][0] - path[idx - 1][0]
            dy = path[idx][1] - path[idx - 1][1]
            return math.atan2(dy, dx)
        return self.yaw

    def _build_reference(
        self, path: list[tuple[float, float]], start_idx: int
    ) -> list[tuple[float, float, float]]:
        """
        Extract N+1 reference states from the path, advancing roughly
        v_max * dt metres per step so the reference moves at a sensible
        forward speed.  If the path is too short, hold the last point.
        """
        ref = []
        # Approximate spacing of reference points on the path
        step_m = max(self.v_max * self.dt, 0.05)
        # Path resolution is ~5 cm (from path_smoother), so step_idx ≈ step_m / 0.05
        # Estimate path resolution from the local segment instead (robust)
        path_res = 0.05
        if len(path) >= 2:
            path_res = max(0.01, dist2d(path[0], path[1]))
        step_idx = max(1, int(round(step_m / path_res)))

        for k in range(self.N + 1):
            idx = min(start_idx + k * step_idx, len(path) - 1)
            rx, ry = path[idx]
            rth = self._estimate_heading(path, idx)
            ref.append((rx, ry, rth))
        return ref

    # ── Control step ─────────────────────────────────────────────────────────

    def _control_step(self):
        if not self.have_odom:
            return

        # ── Startup gating ─────────────────────────────────────────────────
        # Don't move until Gazebo is clearly up: enough odom messages, and
        # enough sim-time elapsed since the first odom message.  This stops
        # the robot from twitching while RViz/Gazebo are still loading.
        if not self.startup_done:
            now = self.get_clock().now().nanoseconds * 1e-9
            elapsed = (now - self.first_odom_time) if self.first_odom_time else 0.0
            if (self.odom_count < self.min_odom_msgs
                    or elapsed < self.startup_wait_sec):
                self._set_cmd(0.0, 0.0)
                self.get_logger().info(
                    f"Waiting for sim startup: odom={self.odom_count}/"
                    f"{self.min_odom_msgs} elapsed={elapsed:.1f}/"
                    f"{self.startup_wait_sec:.1f}s",
                    throttle_duration_sec=1.0,
                )
                return
            self.startup_done = True
            self.get_logger().info("Startup complete — beginning to track path.")

        with self.lock:
            path = list(self.path)
            last_pt = self.last_path_point

        if not path:
            return

        if self.reached_goal:
            self._set_cmd(0.0, 0.0)
            return

        # ── Goal check (against the global goal, not whatever path slice
        # the avoider just published) ─────────────────────────────────────
        if last_pt is not None and dist2d((self.x, self.y), last_pt) < self.goal_tol:
            self._set_cmd(0.0, 0.0)
            self.reached_goal = True
            self.get_logger().info("===========================Goal reached!====================================================================================================")
            return

        # ── Final-resort emergency hard stop ───────────────────────────────
        if self._emergency_stop():
            self._set_cmd(0.0, 0.0)
            if self.debug:
                self.get_logger().warn(
                    f"Emergency stop: obstacle within {self.emg_dist:.2f} m.",
                    throttle_duration_sec=1.0,
                )
            return

        # ── Monotonic progress along the published path ───────────────────
        # Search a bounded window AHEAD of the last progress index only.
        # This guarantees the MPC tracks waypoints in order and never
        # picks a closer point that's actually a previous segment of the
        # path (which is what was happening when the robot was off on a
        # large detour).
        start_idx = self._advance_path_progress(path)

        # Final guard against backing up: skip points that are behind us
        for look_ahead in range(min(5, len(path) - start_idx - 1)):
            idx = start_idx + look_ahead
            dx = path[idx][0] - self.x
            dy = path[idx][1] - self.y
            angle_to_pt = math.atan2(dy, dx)
            angle_diff = abs(wrap(angle_to_pt - self.yaw))
            if angle_diff < math.pi / 2:
                start_idx = idx
                break
        # Persist the advance
        self.path_progress_idx = max(self.path_progress_idx, start_idx)

        ref = self._build_reference(path, start_idx)
        state = (self.x, self.y, self.yaw)

        v_cmd, w_cmd, predicted = self.mpc.solve(
            state, ref, self.prev_v, self.prev_w
        )

        # Publish predicted trajectory and current target for RViz
        self._publish_predicted(predicted)
        self._publish_target(ref[0])

        if self.debug:
            self.get_logger().info(
                f"MPC v={v_cmd:.3f} w={w_cmd:.3f}  "
                f"start={start_idx}/{len(path)} "
                f"ref0=({ref[0][0]:.2f},{ref[0][1]:.2f}) "
                f"goal_dist={dist2d((self.x,self.y), last_pt):.2f}m",
                throttle_duration_sec=0.5,
            )

        self._set_cmd(v_cmd, w_cmd)

    def _publish_predicted(self, predicted: list[tuple[float, float, float]]):
        msg = Path()
        msg.header.frame_id = "odom"
        msg.header.stamp = self.get_clock().now().to_msg()
        for x, y, _th in predicted:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.predicted_pub.publish(msg)

    def _publish_target(self, target: tuple[float, float, float]):
        m = Marker()
        m.header.frame_id = "odom"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "mpc_target"
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(target[0])
        m.pose.position.y = float(target[1])
        m.pose.position.z = 0.25
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.18
        m.color.r = 0.0
        m.color.g = 1.0
        m.color.b = 1.0
        m.color.a = 0.9
        self.target_pub.publish(m)

    def _advance_path_progress(self, path: list[tuple[float, float]]) -> int:
        """
        Return current progress index along `path`.  Only ever moves
        forward.  Searches a bounded window ahead so the MPC visits
        waypoints in order.
        """
        if not path:
            return 0
        self.path_progress_idx = min(self.path_progress_idx, len(path) - 1)
        rob = (self.x, self.y)
        window_end = min(len(path), self.path_progress_idx + 30)
        best_i = self.path_progress_idx
        best_d = dist2d(rob, path[best_i])
        for i in range(self.path_progress_idx + 1, window_end):
            d = dist2d(rob, path[i])
            if d < best_d:
                best_d = d
                best_i = i
        if best_i > self.path_progress_idx:
            self.path_progress_idx = best_i
        return self.path_progress_idx

    def _set_cmd(self, v: float, w: float):
        v = float(np.clip(v, -self.v_max, self.v_max))
        w = float(np.clip(w, -self.w_max,  self.w_max))
        msg = Twist()
        msg.linear.x  = v
        msg.angular.z = w
        self.cmd_pub.publish(msg)
        self.prev_v = v
        self.prev_w = w


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = MPCTrackerNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
