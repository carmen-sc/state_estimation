cmake_minimum_required(VERSION 2.8.3)
project(at_localization)

find_package(catkin REQUIRED COMPONENTS
  rospy
  duckietown_msgs
  image_geometry
)

catkin_package()