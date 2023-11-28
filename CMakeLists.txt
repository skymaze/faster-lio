cmake_minimum_required(VERSION 3.5)
project(faster_lio)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)

# Add compiler flags
add_compile_options(-pthread -fexceptions)

# Add definitions
add_compile_definitions(ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/")

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(livox_ros_driver2 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)

find_package(rosidl_default_generators REQUIRED)


# options
option(WITH_IVOX_NODE_TYPE_PHC "Use PHC instead of default ivox node" OFF)

if (WITH_IVOX_NODE_TYPE_PHC)
    message("USING_IVOX_NODE_TYPE_PHC")
    add_definitions(-DIVOX_NODE_TYPE_PHC)
else ()
    message("USING_IVOX_NODE_TYPE_DEFAULT")
endif()

set(msg_files
  "msg/Pose6D.msg"
)
rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
)
ament_export_dependencies(rosidl_default_runtime)

add_executable(laser_mapping_node src/laser_mapping_node.cpp src/laser_mapping.cpp src/utils.cpp src/pointcloud_preprocess.cpp src/imu_processing.cpp src/use_ikfom.cpp)
target_include_directories(laser_mapping_node PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${PCL_INCLUDE_DIRS}
)
rosidl_get_typesupport_target(cpp_typesupport_target
  ${PROJECT_NAME} rosidl_typesupport_cpp)
  target_link_libraries(laser_mapping_node "${cpp_typesupport_target}")
ament_target_dependencies(laser_mapping_node
    rclcpp
    std_msgs
    sensor_msgs
    nav_msgs
    geometry_msgs
    tf2
    tf2_ros
    pcl_ros
    pcl_conversions
    livox_ros_driver2
)

install(TARGETS
    laser_mapping_node
    DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
    launch
    config
    rviz
    DESTINATION share/${PROJECT_NAME}
)

ament_package()