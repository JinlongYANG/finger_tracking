#define BOOST_SIGNALS_NO_DEPRECATION_WARNING

#include "ros/ros.h"
#include <finger_tracking/finger_tracking_node.hpp>

int main(int argc, char **argv)
{
  // Set up ROS.
  ros::init(argc, argv, "finger_tracking",ros::init_options::NoRosout);
  ros::NodeHandle nh;

    Finger_tracking_Node Finger_tracking_Node(nh);

  // Main loop.
  while (nh.ok())
  {
    ros::spinOnce();
  }
  return 0;
}
