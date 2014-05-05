#ifndef finger_tracking_node_hpp
#define finger_tracking_node_hpp

#include <iostream>
#include <vector>
#include <algorithm>
#include <map>
#include <math.h>
#include <ros/ros.h>
#include <dynamic_reconfigure/server.h>
#include <boost/shared_ptr.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/CameraInfo.h>
#include <geometry_msgs/TransformStamped.h>
#include <nav_msgs/Path.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <image_geometry/pinhole_camera_model.h>
#include <image_geometry/stereo_camera_model.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
//#include "finger_tracking/finger_tracking_Config.h"
#include "finger_tracking/finger_tracking.hpp"

using namespace image_transport;
using namespace sensor_msgs;
using namespace geometry_msgs;
using namespace cv;
using namespace image_geometry;
using namespace pcl;
using namespace Eigen;


class Finger_tracking_Node
{
private:
    image_transport::ImageTransport imageTransport_;
    image_transport::Publisher publisher_;
    message_filters::TimeSynchronizer<Image, Image,CameraInfo,CameraInfo> timeSynchronizer_;
    message_filters::Subscriber<Image>  rgbCameraSubscriber_;
    message_filters::Subscriber<Image> depthCameraSubscriber_;
    message_filters::Subscriber<CameraInfo> rgbCameraInfoSubscriber_;
    message_filters::Subscriber<CameraInfo> depthCameraInfoSubscriber_;
//    dynamic_reconfigure::Server<finger_tracking::finger_tracking_Config> reconfigureServer_;
//    dynamic_reconfigure::Server<finger_tracking::finger_tracking_Config>::CallbackType reconfigureCallback_;

public:
    tf2_ros::TransformBroadcaster transformBroadcaster_;
    tf2_ros::Buffer buffer_;
    tf2_ros::TransformListener transformListener_;
    
    geometry_msgs::TransformStamped transformStamped_;
    
    boost::shared_ptr<Finger_tracking> finger_tracking_;
    cv::Ptr<cv::FeatureDetector> detector_;
    FREAK extractor_;

    ros::Publisher cloud_pub_;
    
    
    
    Finger_tracking_Node(ros::NodeHandle& nh);
    //void updateConfig(finger_tracking::finger_tracking_Config &config, uint32_t level);
    void syncedCallback(const ImageConstPtr& leftImage,const ImageConstPtr& rightImage, const CameraInfoConstPtr& leftInfo, const CameraInfoConstPtr& rightInfo);
    
};
#endif
