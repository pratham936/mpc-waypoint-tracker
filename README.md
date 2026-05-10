# Autonomous Navigation with Regulated Pure Pursuit

A ROS2-based navigation stack implementing path smoothing and regulated pure pursuit control for differential drive robots in constrained environments. [Video]([https://drive.google.com/file/d/17OE4Cf4g9UPwNrmvYiQqMd2OCQQHhMXT/view?usp=sharing](https://drive.google.com/file/d/1CESRBhaxAiyDw4Dapy0Lzjpnj3Ipr7lk/view))

## Overview

This package addresses the challenge of converting discrete waypoints from a global planner into smooth, dynamically-feasible trajectories while ensuring accurate tracking with adaptive velocity regulation.

### Key Features

- **Path Smoothing**: Converts discrete waypoints into continuous trajectories with rounded corners
- **Regulated Pure Pursuit**: Adaptive velocity control based on path curvature and proximity to obstacles
- **Dynamic Obstacle Avoidance**: Real-time laser scan integration for collision prevention
- **Velocity Scaling**: Automatic speed reduction at sharp turns and near goal positions
- **Tight Space Navigation**: Suitable for confined environments with dynamic obstacles

## Problem Statement

Mobile robots typically receive coarse paths consisting of discrete waypoints from global planners. Direct execution of these waypoints results in:
- Jerky, non-smooth motion
- Potential collision risks at sharp corners
- Inefficient velocity profiles
- Poor trajectory tracking accuracy

This implementation solves these issues through:
1. **Path Refinement**: Smoothing discrete waypoints into continuous trajectories
2. **Adaptive Control**: Regulating velocity based on local path geometry and obstacles
3. **Safe Navigation**: Real-time obstacle detection and avoidance

## Architecture

```
┌─────────────────┐
│   Waypoints     │
│   (CSV File)    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  Path Smoother Node     │
│  - Bézier interpolation │
│  - Corner rounding      │
└──────────┬──────────────┘
           │ /path (nav_msgs/Path)
           ▼
┌──────────────────────────────┐
│  Pure Pursuit Controller     │
│  - Lookahead calculation     │
│  - Curvature-based scaling   │
│  - Obstacle detection        │
└──────────┬───────────────────┘
           │ /cmd_vel
           ▼
┌──────────────────┐
│  Differential    │
│  Drive Robot     │
└──────────────────┘
```

## Algorithm Details

### Path Smoothing

The smoother applies Bézier curve interpolation at waypoint corners:
- Detects sharp angles between consecutive segments
- Inserts quadratic Bézier curves with configurable radius
- Maintains path connectivity and resolution
- Preserves start and end points

### Regulated Pure Pursuit

Standard pure pursuit enhanced with:

**Curvature-Based Velocity Regulation**
```
v = v_desired × min(1.0, r_min/r_current)
```
Where `r` is the path curvature radius

**Approach Velocity Scaling**
```
v = v × min(1.0, d_goal/d_scaling)
```
Smoothly decelerates as the robot approaches the goal

**Adaptive Lookahead Distance**
```
L_d = clamp(v × t_lookahead, L_min, L_max)
```
Velocity-scaled lookahead for stable tracking

**Rotate-to-Heading**
- Triggers in-place rotation when heading error exceeds threshold
- Prevents lateral drift at sharp corners

**Obstacle Detection**
- 120° frontal scan coverage (-60° to +60°)
- 0.5m detection threshold
- Immediate stop-and-wait behavior

## Installation

```bash
cd ~/ros2_ws/src
git clone https://github.com/manojkarnekar/smoothing-rpp.git nav
cd ~/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

## Usage

### 1. Launch Simulation Environment

**Terminal 1:**
```bash
export TURTLEBOT3_MODEL=burger
ros2 launch nav sim_bringup.py
```

This launches:
- Gazebo with empty world
- TurtleBot3 model
- Pure pursuit controller
- RViz2 visualization

**Add obstacles in Gazebo** after launch for testing dynamic avoidance.

### 2. Start Path Smoother

**Terminal 2:**
```bash
ros2 run nav smoothing
```

Loads waypoints from CSV, generates smooth trajectory, and publishes:

- Original waypoints to /waypoints (visualized in red in RViz)
- Smoothed trajectory to /path (visualized in green in RViz)

## Configuration

### Waypoint File Format

`waypoints/waypoint.csv`:
```csv
0.0,0.0
1.0,0.0
1.0,1.0
2.0,1.0
```

### Key Parameters

**Path Smoother** (`smoothing` node):
- `path_resolution`: Interpolation resolution (default: 0.1m)
- `path_file`: CSV waypoint file path
- `frame_id`: TF frame for path (default: "odom")

**Pure Pursuit** (`pure_pursuit_tracker` node):
- `desired_linear_vel`: Target velocity (default: 0.3 m/s)
- `lookahead_time`: Lookahead time horizon (default: 1.5s)
- `min_lookahead_distance`: Minimum lookahead (default: 0.3m)
- `max_lookahead_distance`: Maximum lookahead (default: 0.9m)
- `regulated_linear_scaling_min_radius`: Curvature regulation threshold (default: 0.9m)
- `velocity_scaling_distance`: Goal approach distance (default: 0.5m)
- `rotate_to_heading_min_angle`: In-place rotation threshold (default: 0.785 rad)

## Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/path` | nav_msgs/Path | Smoothed trajectory |
| `/waypoints` | nav_msgs/Path | Original waypoints |
| `/cmd_vel` | geometry_msgs/Twist | Velocity commands |
| `/scan` | sensor_msgs/LaserScan | Laser scan data |
| `/odom` | nav_msgs/Odometry | Robot odometry |
| `/pure_pursuit/lookahead_point` | geometry_msgs/PointStamped | Current lookahead point |

## Behavior Characteristics

### Smooth Motion
- Continuous velocity profiles without sudden changes
- Gradual acceleration/deceleration based on curvature
- No oscillations or overshoot at waypoints

### Tight Space Navigation
- Automatic speed reduction in narrow passages
- Stop-and-wait for dynamic obstacles
- Maintains safe clearance from obstacles

### Robustness
- Handles missed TF lookups gracefully
- Path preemption support for replanning
- Goal tolerance checking with configurable thresholds

## Performance

Typical performance metrics:
- Path tracking error: < 0.1m (straight segments)
- Goal reaching accuracy: < 0.2m
- Obstacle detection latency: < 50ms
- Control loop frequency: 30 Hz

## Future Enhancements

- Trajectory optimization with time-optimal velocity profiles
- Model Predictive Control (MPC) for tighter tracking
- Dynamic window approach integration
- Multi-resolution path smoothing
- Recovery behaviors for stuck scenarios

## Dependencies

- ROS2 Humble
- TurtleBot3 packages
- Gazebo Classic
- RViz2
- tf2_ros
- sensor_msgs
- nav_msgs
- geometry_msgs

## License


## Author

Manoj K
