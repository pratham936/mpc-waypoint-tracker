# MPC Waypoint Tracker with Reactive Obstacle Avoidance

A ROS 2 navigation stack for TurtleBot3 that follows a sequence of waypoints
using a model-predictive controller (MPC) and reactively avoids both static
and dynamic obstacles using 2D LiDAR. Tested in Gazebo with RViz
visualization.

## Overview

Three coordinated nodes:

- **`path_smoother`** — loads waypoints from a CSV and fits a cubic B-spline
  through them, resampled at 5 cm to give the MPC a dense reference path.
- **`obstacle_avoider`** — uses 2D LiDAR (`/scan`) to detect obstacles in the
  path ahead. When blocked, plans a clockwise three-waypoint arc detour
  around the obstacle and merges back onto the global path past it.
  Re-plans only when the planned arc itself becomes blocked, so the robot
  follows a stable trajectory rather than wobbling.
- **`mpc_tracker`** — solves a 10-step (1 s) optimisation each control cycle
  using CasADi + IPOPT with a unicycle kinematic model. Tracks whichever
  path the avoider currently publishes (global or detour).

```
   waypoint.csv
        │
        ▼
  path_smoother ──/path──► obstacle_avoider ──/path_local──► mpc_tracker ──/cmd_vel
                                   ▲
                                /scan
```

## Dependencies

- ROS 2 (Humble or newer)
- TurtleBot3 packages (`turtlebot3_gazebo`, `turtlebot3_msgs`)
- Python: `numpy`, `scipy`, `casadi`

```bash
sudo apt install ros-$ROS_DISTRO-turtlebot3-gazebo
pip install numpy scipy casadi
```

## Build

Place this package inside your workspace's `src/` folder, then:

```bash
cd ~/ros2_ws
colcon build --packages-select nav
source install/setup.bash
```

## Run

```bash
export TURTLEBOT3_MODEL=burger
ros2 launch nav sim_bringup.py
```

This opens Gazebo (TurtleBot3 in an empty world), RViz with the navigation
visualization config, and starts all three navigation nodes. The robot
loads the waypoints from `waypoints/waypoint.csv` and begins tracking
them.

To test obstacle avoidance, drop a Gazebo box (Insert → Construction Cone
or any box) in front of the robot mid-mission. The robot should plan a
clockwise detour around it and rejoin the original path automatically.

## Waypoints

Waypoints are loaded from `waypoints/waypoint.csv` as comma-separated
`x,y` pairs in the `odom` frame. Edit the file to change the route — the
node will pick up the new waypoints on the next launch.

## RViz colour scheme

| Display | Topic | What it represents |
|---|---|---|
| Green line | `/path` | Smoothed global path through CSV waypoints |
| Blue line | `/path_local` | Path the MPC currently tracks (global or detour) |
| Orange line | `/viz/detour_path` | Active detour arc around an obstacle |
| Pink line (off by default) | `/viz/mpc_predicted` | MPC's 1-second predicted trajectory |
| Cyan sphere | `/viz/mpc_target` | MPC's current reference target this cycle |
| Cyan + WP labels | `/viz/csv_waypoints` | Original waypoints from `waypoint.csv` |
| Orange + D labels | `/viz/detour_waypoints` | Transient detour arc waypoints |
| White dots | `/scan` | Raw 2D LiDAR returns |
| Red dots | `/viz/obstacles` | Filtered obstacle points used by the avoider |

## Tuning

All tunable parameters live in `launch/sim_bringup.py` and can be edited
without touching the node code. Useful knobs:

- `detour_offset` — lateral clearance of the detour arc (m)
- `merge_forward_margin` — how far past the obstacle the arc rejoins the path
- `arc_forward_stretch` — multiplier on the arc's forward extent
- `post_detour_cooldown_m` — suppress new detours for this distance after one ends
- `clockwise_only` — always dodge to the right of the path; set False for auto side-pick
- `v_max`, `w_max`, `mpc_horizon`, `q_pos`, `r_v` etc. — MPC weights and limits

## License

MIT
