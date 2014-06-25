cmake_minimum_required(VERSION 2.8.3)
project(finger_tracking)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages

find_package(catkin REQUIRED COMPONENTS cv_bridge dynamic_reconfigure image_transport image_geometry sensor_msgs roscpp pcl_ros tf2_ros)

find_package(OpenCV REQUIRED)
find_package(PCL 1.6 REQUIRED)

generate_dynamic_reconfigure_options(
  cfg/finger_tracking_Config.cfg
)

## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
   INCLUDE_DIRS include
#  LIBRARIES pixel_features
#  CATKIN_DEPENDS cv_bridge dynamic_reconfigure image_transport opencv roscpp
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
include_directories(include
        ${catkin_INCLUDE_DIRS}
	${PCL_INCLUDE_DIRS}
)

include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS} )


## Declare a cpp executable
add_executable(finger_tracking
	src/finger_tracking_main.cpp
	src/finger_tracking_node.cpp
	src/handkp_leap_msg.cpp
	src/Hand_XYZRGB.cpp
	src/articulate_HandModel_XYZRGB.cpp
	include/GCoptimization/graph.cpp
	include/GCoptimization/GCoptimization.cpp
	include/GCoptimization/maxflow.cpp
	include/GCoptimization/LinkedBlockList.cpp
)


## Add dependencies to the executable
add_dependencies(finger_tracking ${PROJECT_NAME}_gencfg)


## Specify libraries to link a library or executable target against
target_link_libraries(finger_tracking 	${catkin_LIBRARIES}
   	${OpenCV_LIBRARIES}
	${PCL_LIBRARIES}
)
