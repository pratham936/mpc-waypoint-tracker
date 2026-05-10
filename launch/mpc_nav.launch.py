#!/usr/bin/env python3
"""
launch/mpc_nav.launch.py
========================
Launches all three nodes in the MPC navigation stack:

  1. path_smoother_node  – reads CSV, smooths, publishes /path
  2. obstacle_avoider_node – watches /scan, modifies /path → /path_local
  3. mpc_tracker_node    – runs MPC on /path_local, publishes /cmd_vel

Usage (from workspace root):
    ros2 launch nav mpc_nav.launch.py path_file:=src/nav/waypoints/waypoint.csv
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    path_file_arg = DeclareLaunchArgument(
        "path_file",
        #default_value="src/nav/waypoints/waypoint.csv",
        default_value=os.path.join(get_package_share_directory('nav'), 'waypoints', 'waypoint.csv'),
        description="Path to the waypoints CSV file",
    )
    debug_arg = DeclareLaunchArgument(
        "debug", default_value="true", description="Enable debug logging"
    )
    frame_id_arg = DeclareLaunchArgument(
        "frame_id", default_value="odom", description="World frame ID"
    )

    path_file = LaunchConfiguration("path_file")
    debug     = LaunchConfiguration("debug")
    frame_id  = LaunchConfiguration("frame_id")

    smoother = Node(
        package="nav",
        executable="path_smoother",
        name="path_smoother_node",
        output="screen",
        parameters=[{
            "path_file":       path_file,
            "frame_id":        frame_id,
            "path_resolution": 0.05,
            "smoothing_factor": 0.0,
            "spline_degree":   3,
        }],
    )

    avoider = Node(
        package="nav",
        executable="obstacle_avoider",
        name="obstacle_avoider_node",
        output="screen",
        parameters=[{
            "check_horizon":      2.0,
            "obstacle_clearance": 0.55,
            "detour_offset":      0.70,
            "control_hz":         10.0,
            "robot_radius":       0.25,
            "frame_id":           frame_id,
            "lookahead_check_pts": 20,
        }],
    )

    tracker = Node(
        package="nav",
        executable="mpc_tracker",
        name="mpc_tracker_node",
        output="screen",
        parameters=[{
            "control_hz":           10.0,
            "mpc_horizon":          10,
            "mpc_dt":               0.1,
            "v_max":                0.5,
            "w_max":                0.8,
            "dv_max":               0.15,
            "dw_max":               0.30,
            "goal_tolerance":       0.20,
            "q_pos":                5.0,
            "q_yaw":                1.0,
            "r_v":                  0.1,
            "r_w":                  0.2,
            "r_dv":                 0.5,
            "r_dw":                 0.5,
            "q_terminal":          10.0,
            "emergency_stop_dist":  0.40,
            "emergency_stop_angle": 60.0,
            "debug":                debug,
        }],
    )

    return LaunchDescription([
        path_file_arg,
        debug_arg,
        frame_id_arg,
        smoother,
        avoider,
        tracker,
    ])
