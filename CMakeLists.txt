cmake_minimum_required(VERSION 3.8)
project(nav)

# author: Pratham
# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(PkgConfig)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rclpy REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

ament_python_install_package(${PROJECT_NAME})

set(dependencies
rclcpp
rclcpp_action
nav_msgs
geometry_msgs
sensor_msgs
std_msgs
tf2
tf2_ros
tf2_geometry_msgs
)

include_directories(
  include
)

install(
  DIRECTORY
    launch config waypoints nav src
  DESTINATION
    share/${PROJECT_NAME}
)

# add_executable(serial_port src/serial_port.cpp)
add_executable(pure_pursuit src/pure_pursuit.cpp)
ament_target_dependencies(pure_pursuit ${dependencies})

add_executable(smoothing src/smoothing.cpp)
ament_target_dependencies(smoothing ${dependencies})

install(TARGETS
    pure_pursuit
    smoothing
    DESTINATION lib/${PROJECT_NAME})

install(PROGRAMS
    scripts/path_smoother.py
    scripts/obstacle_avoider.py
    scripts/mpc_tracker.py
    DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch/
    DESTINATION share/${PROJECT_NAME}/launch)

install(DIRECTORY waypoints/
  DESTINATION share/${PROJECT_NAME}/waypoints)

ament_package()
