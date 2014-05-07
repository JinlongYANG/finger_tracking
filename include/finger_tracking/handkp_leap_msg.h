#ifndef HandKeyPoints_MSG_H
#define HandKeyPointsP_MSG_H
#include <leap_msgs/Leap.h>
#include <opencv2/core/core.hpp>

using namespace cv;

class HandKeyPoints
{
public:
    HandKeyPoints();

    void set_Leap_Msg(const leap_msgs::Leap::ConstPtr& msg);

    void put_into_cloud();

    void Clear();

    int64_t frame_id;
    int64_t time_stamp;
    int16_t hands_count;
    int16_t fingers_count;

    std::vector<Point3d> hand_position;
    std::vector<double> hand_orientation;
    std::vector<Point3d> fingertip_position;
    std::vector<Point3d> finger_direction;
    std::vector<Point3d> fingertip_velocity;
    std::vector<Point2d> finger_shape;
    std::vector<int16_t> finger_ids;

private:

};

#endif // OBJ_LEAP_MSG_H

