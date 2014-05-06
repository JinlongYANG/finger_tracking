#include "handkp_leap_msg.h"


HandKeyPoints::HandKeyPoints()
{
    frame_id=0;
    time_stamp=0;
    hands_count=0;
    fingers_count=0;

}



void Leap_Msg::set_Leap_Msg(const leap_msgs::Leap::ConstPtr& msg)       //2nd & 3rd argument are ids from previous frame
{
    frame_id = msg->leap_frame_id;
    time_stamp = msg->leap_time_stamp;
    hands_count = msg->hands.size();
    fingers_count = msg->fingers.size();

    for(int i=0;i< msg->hands.size(); i++)
    {

        for(int j=0;j< msg->fingers.size(); j++)
        {
            Point3d pt3d;
            pt3d.x = msg->finger.at(j).pose.position.x;
            pt3d.y = msg->finger.at(j).pose.position.y;
            pt3d.z = msg->finger.at(j).pose.position.z;
            fingertip_position.push_back(pt3d);

            pt3d.x = msg->finger.at(j).pose.direction.x;
            pt3d.y = msg->finger.at(j).pose.direction.y;
            pt3d.z = msg->finger.at(j).pose.direction.z;
            finger_direction.push_back(pt3d);

            pt3d.x = msg->finger.at(j).pose.velocity.x;
            pt3d.y = msg->finger.at(j).pose.velocity.y;
            pt3d.z = msg->finger.at(j).pose.velocity.z;
            fingertip_velocity.push_back(pt3d);

            Point2d pt2d;
            pt2d.x = msg->finger.at(j).length;
            pt2d.y = msg->finger.at(j).width;
            finger_shape.push_back(pt2d);

        }

    }
}


void Obj_Leap_Msg::Clear()
{
    hands_count=0;
    fingers_count=0;
    fingertip_position;
    finger_direction.clear();
    fingertip_velocity.clear();
    finger_shape.clear();
}
