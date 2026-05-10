import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():

    pkg_share = get_package_share_directory('nav')

    turtlebot3_gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('turtlebot3_gazebo'),
                'launch',
                'empty_world.launch.py'
            ])
        ]),
        launch_arguments={
            'use_sim_time': 'true',
            # Force spawn at the world origin so the path's index 0
            # matches the robot's initial position.
            'x_pose': '0.0',
            'y_pose': '0.0',
        }.items()
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=[
            '-d',
            os.path.join(pkg_share, 'config', 'sim.rviz')
        ],
        parameters=[{'use_sim_time': True}]
    )

    path_smoother = Node(
        package='nav',
        executable='path_smoother.py',
        name='path_smoother_node',
        parameters=[{
            'use_sim_time':     True,
            'path_file':        os.path.join(pkg_share, 'waypoints', 'waypoint.csv'),
            'frame_id':         'odom',
            'path_resolution':  0.05,
            'smoothing_factor': 0.0,
            'spline_degree':    3,
            'publish_delay_sec': 2.0,
        }]
    )

    obstacle_avoider = Node(
        package='nav',
        executable='obstacle_avoider.py',
        name='obstacle_avoider_node',
        parameters=[{
            'use_sim_time':         True,
            'check_horizon':        3.0,
            'obstacle_clearance':   0.40,   # buffer past robot radius
            'detour_offset':        1.10,   # lateral detour size (m) — bigger = more clearance
            'control_hz':          10.0,
            'robot_radius':         0.25,
            'frame_id':            'odom',
            'lookahead_check_pts':  30,     # 1.5 m at 5 cm spacing → see obstacles early
            'scan_max_range':       3.0,
            'stop_dist':            0.18,   # only e-stop when very close
            'recovery_hold_sec':    1.0,
            'min_block_hits':       2,
            'clockwise_only':       True,   # always dodge to the right of the path
            # Merge-back tuning: detour ends further past the obstacle
            # so the robot doesn't immediately trigger a second detour.
            'merge_forward_margin': 1.20,   # m past obstacle's far edge before rejoin
            'arc_forward_stretch':  1.40,   # multiplier on wp3 forward distance
            'post_detour_cooldown_m': 1.50, # suppress new detours for 1.5m after one ends
        }]
    )

    mpc_tracker = Node(
        package='nav',
        executable='mpc_tracker.py',
        name='mpc_tracker_node',
        parameters=[{
            'use_sim_time':         True,
            'control_hz':          10.0,
            'mpc_horizon':         10,
            'mpc_dt':               0.1,
            'v_max':                0.5,
            'w_max':                0.8,
            'dv_max':               0.15,
            'dw_max':               0.30,
            'goal_tolerance':       0.20,
            'q_pos':                5.0,
            'q_yaw':                1.0,
            'r_v':                  0.1,
            'r_w':                  0.2,
            'r_dv':                 0.5,
            'r_dw':                 0.5,
            'q_terminal':          10.0,
            'emergency_stop_dist':  0.18,
            'emergency_stop_angle': 20.0,
            # Startup gating: wait this long (sim-time seconds) AND this
            # many odom messages before commanding any motion.  Stops the
            # robot from twitching while Gazebo is still loading.
            'startup_wait_sec':     5.0,
            'min_odom_msgs':        20,
            'debug':                True,
        }]
    )

    return LaunchDescription([
        turtlebot3_gazebo,
        rviz,
        path_smoother,
        obstacle_avoider,
        mpc_tracker,
    ])
