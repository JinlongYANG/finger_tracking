#ifndef Hand_XYZRGB_H
#define Hand_XYZRGB_H

#include <opencv2/core/core.hpp>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>

using namespace cv;

class Hand_XYZRGB
{
public:
    Hand_XYZRGB();

    pcl::PointXYZRGB palm_center;
    std::vector<pcl::PointXYZRGB> fingertip_position;
    std::vector<pcl::PointXYZRGB> finger_direction;

private:


};

#endif // Hand_XYZRGB_H

