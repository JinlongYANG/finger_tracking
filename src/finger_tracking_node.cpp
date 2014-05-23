#define BOOST_SIGNALS_NO_DEPRECATION_WARNING
#include "finger_tracking/finger_tracking_node.hpp"
#include "finger_tracking/handkp_leap_msg.h"
#include "finger_tracking/Hand_XYZRGB.h"
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <math.h>

#define PI 3.14159265


using namespace image_transport;
using namespace sensor_msgs;
using namespace geometry_msgs;
using namespace cv;
using namespace image_geometry;
using namespace pcl;
using namespace Eigen;
using namespace std;
//using namespace tf2;

float half_palm_length = 0.07;
float finger_length = 0.07;
int resolution = 2;
//30cm *30cm for hand
int imageSize = 300/resolution;




Finger_tracking_Node::Finger_tracking_Node(ros::NodeHandle& nh):
    imageTransport_(nh),
    timeSynchronizer_(10)
  //reconfigureServer_(ros::NodeHandle(nh,"finger_tracking")),
  //transformListener_(buffer_, true)
  //reconfigureCallback_(boost::bind(&finger_tracking_Node::updateConfig, this, _1, _2))
{

    hand_kp_Subscriber_.subscribe(nh, "/Hand_kp_cl", 5);;
    hand_Subscriber_.subscribe(nh, "/Hand_pcl", 5);

    timeSynchronizer_.connectInput(hand_kp_Subscriber_, hand_Subscriber_);

    depthpublisher_ = imageTransport_.advertise("DepthImage", 1);
    segmentpublisher_ = imageTransport_.advertise("Segment", 1);

    articulatePublisher_ = nh.advertise<sensor_msgs::PointCloud2>("Articulate",1);

    timeSynchronizer_.registerCallback(boost::bind(&Finger_tracking_Node::syncedCallback, this, _1, _2));
    //reconfigureServer_.setCallback(reconfigureCallback_);


}

//void  SlamNode::updateConfig(pixel_slam::slamConfig &config, uint32_t level){
//    slam_.reset(new Slam(config.min_depth,config.max_depth,config.line_requirement));
//    line_requirement_=config.line_requirement;
//    if(stereoCameraModel_.initialized()){
//        min_disparity_=stereoCameraModel_.getDisparity(config.max_depth);
//        max_disparity_=stereoCameraModel_.getDisparity(config.min_depth);
//    }
//}

void Finger_tracking_Node::syncedCallback(const PointCloud2ConstPtr& hand_kp_pter, const PointCloud2ConstPtr& hand_pter){


    //cv_bridge::CvImagePtr cvpointer_rgbFrame, cvpointer_depthFrame;



    pcl::PointCloud<pcl::PointXYZRGB> msg_pcl, handcloud, hand1_kp;

    ROS_INFO("Callback begins");

    try
    {
        int seq = hand_kp_pter->header.seq;

        fromROSMsg(*hand_kp_pter, hand1_kp);
        fromROSMsg(*hand_pter, msg_pcl);

        std::cout<<"Hand Keypoint size: "<<hand1_kp.size()<<endl;
        std::cout<<"Hand Cloud size: "<<msg_pcl.size()<<endl;

        Mat DepthMat;
        DepthMat = Mat::zeros(imageSize,imageSize,CV_8U);

        for (size_t i = 0; i < msg_pcl.points.size (); ++i){
            if( (abs(msg_pcl.points[i].x) < 0.2
                 && abs(msg_pcl.points[i].y) < 0.2
                 && abs(msg_pcl.points[i].z) < 0.2)
                    &&((abs(msg_pcl.points[i].x) < half_palm_length
                        && abs(msg_pcl.points[i].y) < half_palm_length
                        && abs(msg_pcl.points[i].z) < half_palm_length)||
                       (abs(msg_pcl.points[i].x - hand1_kp.points[1].x) < finger_length
                        && abs(msg_pcl.points[i].y - hand1_kp.points[1].y) < finger_length
                        && abs(msg_pcl.points[i].z - hand1_kp.points[1].z) < finger_length)||
                       (abs(msg_pcl.points[i].x - hand1_kp.points[2].x) < finger_length
                        && abs(msg_pcl.points[i].y - hand1_kp.points[2].y) < finger_length
                        && abs(msg_pcl.points[i].z - hand1_kp.points[2].z) < finger_length)||
                       (abs(msg_pcl.points[i].x - hand1_kp.points[3].x) < finger_length
                        && abs(msg_pcl.points[i].y - hand1_kp.points[3].y) < finger_length
                        && abs(msg_pcl.points[i].z - hand1_kp.points[3].z) < finger_length)||
                       (abs(msg_pcl.points[i].x - hand1_kp.points[4].x) < finger_length
                        && abs(msg_pcl.points[i].y - hand1_kp.points[4].y) < finger_length
                        && abs(msg_pcl.points[i].z - hand1_kp.points[4].z) < finger_length)||
                       (abs(msg_pcl.points[i].x - hand1_kp.points[5].x) < finger_length
                        && abs(msg_pcl.points[i].y - hand1_kp.points[5].y) < finger_length
                        && abs(msg_pcl.points[i].z - hand1_kp.points[5].z) < finger_length))
                    ){
                handcloud.push_back(msg_pcl.points[i]);
                int x = int(msg_pcl.points[i].x * 1000);
                int y = int(msg_pcl.points[i].y * 1000);
                int z = int(msg_pcl.points[i].z * 1000);
                int row = y/resolution + imageSize/2;
                int col = x/resolution + imageSize/2;
                int depth = z/resolution + imageSize/2;
                //std::cout<<"row col depth:"<<row<<" "<<col<<" "<<depth<<endl;
                if( row < imageSize && col < imageSize){
                    DepthMat.at<unsigned char>(row, col) = depth;
                }

            }

        }      
        std::cout<<"Hand Cloud size: "<<handcloud.size()<<endl;

        Mat markers= Mat::zeros(imageSize,imageSize,CV_32SC1);
        Mat SegmentMat = Mat::zeros(imageSize,imageSize,CV_8UC3);
        segment(DepthMat, SegmentMat, hand1_kp, resolution, markers);

        sensor_msgs::PointCloud2 cloud_msg;
        toROSMsg(handcloud,cloud_msg);
        cloud_msg.header.frame_id=hand_kp_pter->header.frame_id;
        cloud_msg.header.stamp = hand_kp_pter->header.stamp;
        articulatePublisher_.publish(cloud_msg);

        cv_bridge::CvImage depthImage_msg;
        depthImage_msg.encoding = sensor_msgs::image_encodings::MONO8;
        depthImage_msg.image    = DepthMat;
        depthImage_msg.header.seq = seq;
        depthImage_msg.header.frame_id = seq;
        depthImage_msg.header.stamp =  hand_kp_pter->header.stamp;
        depthpublisher_.publish(depthImage_msg.toImageMsg());

        cv_bridge::CvImage segmentImage_msg;
        segmentImage_msg.encoding = sensor_msgs::image_encodings::TYPE_8UC3;
        segmentImage_msg.image    = SegmentMat;
        segmentImage_msg.header.seq = seq;
        segmentImage_msg.header.frame_id = seq;
        segmentImage_msg.header.stamp =  hand_kp_pter->header.stamp;
        segmentpublisher_.publish(segmentImage_msg.toImageMsg());


        //        /*******************   get color image and depth image    *******************/
        //        cvpointer_rgbFrame = cv_bridge::toCvCopy(cvpointer_rgbImage);
        //        cvpointer_depthFrame = cv_bridge::toCvCopy(cvpointer_depthImage);

        //        ROS_INFO("current image seq: %d ",cvpointer_rgbInfo->header.seq);
        //        BGRImage=cvpointer_rgbFrame->image;
        //        cvtColor( BGRImage, BGRImage, CV_RGB2BGR);
        //        DepthMat=cvpointer_depthFrame->image;
        //        /*maximum depth: 3m*/
        //        DepthImage = DepthMat*0.33;

        //        /*******************   read in the leapmotion data   *******************/
        //        hand_kpt.set_Leap_Msg(ptr_leap);
        //        if(hand_kpt.hands_count != 0){
        //            std::cout<< hand_kpt.hands_count<<std::endl;
        //            hand1_XYZRGB.palm_center.x = hand_kpt.hand_position.at(0).x/1000.0;
        //            hand1_XYZRGB.palm_center.y = hand_kpt.hand_position.at(0).y/1000.0;
        //            hand1_XYZRGB.palm_center.z = hand_kpt.hand_position.at(0).z/1000.0;
        //            uint8_t r1 = 0, g1 = 255, b1 = 0;
        //            uint32_t rgb1 = ((uint32_t)r1 << 16 | (uint32_t)g1 << 8 | (uint32_t)b1);
        //            hand1_XYZRGB.palm_center.rgb = *reinterpret_cast<float*>(&rgb1);
        //            std::cout<<"hand1: palm center: "<<hand1_XYZRGB.palm_center.x<<" "<<hand1_XYZRGB.palm_center.y << " " << hand1_XYZRGB.palm_center.z<<std::endl;

        //            pcl::PointXYZRGB h1_fingertips;
        //            for(size_t i = 0; i < hand_kpt.fingertip_position.size ()/* && i < 5*/; ++i){
        //                h1_fingertips.x = hand_kpt.fingertip_position.at(i).x/1000.0;
        //                h1_fingertips.y = hand_kpt.fingertip_position.at(i).y/1000.0;
        //                h1_fingertips.z = hand_kpt.fingertip_position.at(i).z/1000.0;
        //                uint8_t rf = 0, gf = 150, bf = 150;
        //                uint32_t rgbf = ((uint32_t)rf << 16 | (uint32_t)gf << 8 | (uint32_t)bf);
        //                h1_fingertips.rgb = *reinterpret_cast<float*>(&rgbf);
        //                hand1_XYZRGB.fingertip_position.push_back(h1_fingertips);

        //            }

        //            if(hand_kpt.hands_count > 1){
        //                hand2_XYZRGB.palm_center.x = hand_kpt.hand_position.at(1).x/1000.0;
        //                hand2_XYZRGB.palm_center.y = hand_kpt.hand_position.at(1).y/1000.0;
        //                hand2_XYZRGB.palm_center.z = hand_kpt.hand_position.at(1).z/1000.0;
        //                uint8_t r2 = 0, g2 = 255, b2 = 0;
        //                uint32_t rgb2 = ((uint32_t)r2 << 16 | (uint32_t)g2 << 8 | (uint32_t)b2);
        //                hand2_XYZRGB.palm_center.rgb = *reinterpret_cast<float*>(&rgb2);
        //                std::cout<<"hand2: palm center: "<<hand2_XYZRGB.palm_center.x<<" "<<hand2_XYZRGB.palm_center.y << " " << hand2_XYZRGB.palm_center.z<<std::endl;
        //            }
        //        }

        //        /*******************   Leap motion to Xtion coordinate transform   *******************/

        //        if(hand_kpt.hands_count != 0){
        //            PointXYZRGB after_transform;
        //            after_transform.x = hand1_XYZRGB.palm_center.x*0.367157+hand1_XYZRGB.palm_center.y*0.29586+hand1_XYZRGB.palm_center.z*0.881852-0.0140586;
        //            after_transform.y = hand1_XYZRGB.palm_center.x*0.299282+hand1_XYZRGB.palm_center.y*(-0.935226)+hand1_XYZRGB.palm_center.z*0.189161+0.284407;
        //            after_transform.z = hand1_XYZRGB.palm_center.x*0.880696+hand1_XYZRGB.palm_center.y*0.194471+hand1_XYZRGB.palm_center.z*(-0.431921)+0.591624;
        //            after_transform.rgb = hand1_XYZRGB.palm_center.rgb;
        //            hand1_kpt.push_back(after_transform);

        //            std::cout<<"Palm center: "<<hand1_kpt.at(0).x<<" "<<hand1_kpt.at(0).y << " " << hand1_kpt.at(0).z<<std::endl;
        //            for(size_t i = 0; i < hand1_XYZRGB.fingertip_position.size(); ++i){
        //                PointXYZRGB finger_after_transform;
        //                finger_after_transform.x = hand1_XYZRGB.fingertip_position.at(i).x*0.367157+hand1_XYZRGB.fingertip_position.at(i).y*0.29586+hand1_XYZRGB.fingertip_position.at(i).z*0.881852-0.0140586;
        //                finger_after_transform.y = hand1_XYZRGB.fingertip_position.at(i).x*0.299282+hand1_XYZRGB.fingertip_position.at(i).y*(-0.935226)+hand1_XYZRGB.fingertip_position.at(i).z*0.189161+0.284407;
        //                finger_after_transform.z = hand1_XYZRGB.fingertip_position.at(i).x*0.880696+hand1_XYZRGB.fingertip_position.at(i).y*0.194471+hand1_XYZRGB.fingertip_position.at(i).z*(-0.431921)+0.591624;
        //                finger_after_transform.rgb = hand1_XYZRGB.fingertip_position.at(i).rgb;
        //                hand1_kpt.push_back(finger_after_transform);
        //                std::cout<<"finger "<<i<<": "<<hand1_kpt.at(i).x<<" "<<hand1_kpt.at(i).y << " " << hand1_kpt.at(i).z<<std::endl;
        //            }
        //            std::cout<<"Size: "<<hand1_kpt.size()<<std::endl;

        //            /*******************   get point cloud    *******************/
        //            fromROSMsg(*pclpointer_pointCloud2, msg_pcl);

        //            std::cout<<msg_pcl.points.size()<<std::endl;}

        //        /*******************   segment hand from the point cloud   *******************/
        //        pcl::PointXYZRGB p;
        //        uint8_t r = 255, g = r, b = r;
        //        uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);

        //        for (size_t i = 0; i < msg_pcl.points.size (); ++i){
        //            if(msg_pcl.points[i].z < max_depth && msg_pcl.points[i].z > min_depth){
        //                p.rgb = *reinterpret_cast<float*>(&rgb);
        //                p.x = msg_pcl.points[i].x;
        //                p.y = msg_pcl.points[i].y;
        //                p.z = msg_pcl.points[i].z;
        //                handcloud.push_back(p);
        //            }
        //        }


        //        /*******************   Convert the CvImage to a ROS image message and publish it to topics.   *******************/
        //        cv_bridge::CvImage bgrImage_msg;
        //        bgrImage_msg.encoding = sensor_msgs::image_encodings::BGR8;
        //        bgrImage_msg.image    = BGRImage;
        //        bgrImage_msg.header.seq = seq;
        //        bgrImage_msg.header.frame_id = seq;
        //        bgrImage_msg.header.stamp = ros::Time::now();
        //        bgrImagePublisher_.publish(bgrImage_msg.toImageMsg());

        //        cv_bridge::CvImage depthImage_msg;
        //        depthImage_msg.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
        //        depthImage_msg.image    = DepthImage;
        //        depthImage_msg.header.seq = seq;
        //        depthImage_msg.header.frame_id = seq;
        //        depthImage_msg.header.stamp = ros::Time::now();
        //        depthImagePublisher_.publish(depthImage_msg.toImageMsg());

        //        /*******************   publish pointCloud   *******************/
        //        sensor_msgs::PointCloud2 cloud_msg;
        //        toROSMsg(handcloud,cloud_msg);
        //        cloud_msg.header.frame_id=cvpointer_depthInfo->header.frame_id;
        //        cloud_pub_.publish(cloud_msg);

        //        sensor_msgs::PointCloud2 hand1_kpt_msg;
        //        toROSMsg(hand1_kpt,hand1_kpt_msg);
        //        hand1_kpt_msg.header.frame_id=cvpointer_depthInfo->header.frame_id;
        //        hkp_cloud_pub_.publish(hand1_kpt_msg);

        //        /*******************   clear data   *******************/
        //        hand_kpt.Clear();


        ROS_INFO("One callback done");


    }
    catch (std::exception& e)
    {
        //if there is an error during conversion, display it
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

}


