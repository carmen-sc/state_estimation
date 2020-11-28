cmake_minimum_required(VERSION 2.8.3)
project(fused_localization)

find_package(catkin REQUIRED COMPONENTS
  rospy
  duckietown_msgs
  message_generation
  geometry_msgs
)

add_service_files(FILES
  SpecialService.srv
)

generate_messages( DEPENDENCIES
  geometry_msgs
)

catkin_package()