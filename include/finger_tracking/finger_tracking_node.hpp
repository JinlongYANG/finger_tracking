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
#include <message_filters/sync_policies/approximate_time.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/CameraInfo.h>
#include <geometry_msgs/TransformStamped.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
//#include <opencv2/core/eigen.hpp>
//#include <opencv2/contrib/contrib.hpp>
#include <image_geometry/pinhole_camera_model.h>
//#include <image_geometry/stereo_camera_model.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
//#include <tf2_ros/transform_broadcaster.h>
//#include <tf2_ros/transform_listener.h>
//#include "finger_tracking/finger_tracking_Config.h"
#include "finger_tracking/finger_tracking.hpp"
#include <leap_msgs/Leap.h>
#include <cstdio>



using namespace image_transport;
using namespace sensor_msgs;
using namespace geometry_msgs;
using namespace cv;
using namespace image_geometry;
using namespace pcl;
using namespace Eigen;
using namespace std;


class Finger_tracking_Node
{
private:
    image_transport::ImageTransport imageTransport_;
    image_transport::Publisher depthpublisher_;
    image_transport::Publisher segmentpublisher_;

    message_filters::TimeSynchronizer<PointCloud2, PointCloud2> timeSynchronizer_;
    message_filters::Subscriber<PointCloud2> hand_kp_Subscriber_;
    message_filters::Subscriber<PointCloud2> hand_Subscriber_;

    //    dynamic_reconfigure::Server<finger_tracking::finger_tracking_Config> reconfigureServer_;
    //    dynamic_reconfigure::Server<finger_tracking::finger_tracking_Config>::CallbackType reconfigureCallback_;

public:
    //    tf2_ros::TransformBroadcaster transformBroadcaster_;
    //    tf2_ros::Buffer buffer_;
    //    tf2_ros::TransformListener transformListener_;
    
    ros::Publisher articulatePublisher_;
    boost::shared_ptr<Finger_tracking> finger_tracking_;


    Finger_tracking_Node(ros::NodeHandle& nh);
    //void updateConfig(finger_tracking::finger_tracking_Config &config, uint32_t level);
    void syncedCallback(const PointCloud2ConstPtr& hand_kp_pter, const PointCloud2ConstPtr& hand_pter);

    void segment(Mat source, Mat & output, pcl::PointCloud<pcl::PointXYZRGB> hand_kp, int resolution, Mat & markers){
        int palmcenter_row = source.rows/2;
        int palmcenter_col = palmcenter_row;
        int palm_radius = 45/resolution;
        std::cout<<"palm_radius: "<<palm_radius<<std::endl;
        int count = 0;
        float temp_row = 0, temp_col = 0;
        //Step 1: local center of palm(initialization is in the center of image)
        for(int i = 0; i < 9; i++){
            for(int row_shift = -palm_radius; row_shift <= palm_radius; row_shift++){
                for(int col_shift = -palm_radius; col_shift <= palm_radius; col_shift++){
                    if(row_shift*row_shift+col_shift*col_shift <= palm_radius*palm_radius
                            &&
                            source.at<unsigned char>(palmcenter_row+row_shift, palmcenter_col+col_shift)!=0){
                        temp_row += palmcenter_row+row_shift;
                        temp_col += palmcenter_col+col_shift;
                        count++;
                    }
                }
            }
            palmcenter_row = temp_row/count;
            palmcenter_col = temp_col/count;
            temp_row = 0;
            temp_col = 0;
            count = 0;
        }
        //Step 2: find the palm, set it as gree
        std::cout<<"palm_center: "<<palmcenter_row<<", "<<palmcenter_col<<std::endl;

        for(int row_shift = -palm_radius; row_shift <= palm_radius; row_shift++){
            for(int col_shift = -palm_radius; col_shift <= palm_radius; col_shift++){
                if(row_shift*row_shift+col_shift*col_shift <= palm_radius*palm_radius
                        &&
                        source.at<unsigned char>(palmcenter_row+row_shift, palmcenter_col+col_shift)!=0){
                    source.at<unsigned char>(palmcenter_row+row_shift, palmcenter_col+col_shift) = 0;
                    output.at<unsigned char>(palmcenter_row+row_shift, (palmcenter_col+col_shift)*3+1) = 128;
                }
            }
        }

        markers.at<int>(palmcenter_row, palmcenter_col) = 255;
        output.at<unsigned char>(palmcenter_row, palmcenter_col*3+1) = 255;

        //Step 3: find fingers


        //Step 3.1 read and locate fingertips positions
        int finger_positions[3][5] = {};
        for(int i = 1; i < hand_kp.points.size (); ++i){
            //row
            finger_positions[0][i-1] = int(hand_kp.points[i].y * 1000)/resolution + source.rows/2;
            //col
            finger_positions[1][i-1] = int(hand_kp.points[i].x * 1000)/resolution + source.cols/2;
            //value
            finger_positions[2][i-1] = int(hand_kp.points[i].z * 1000)/resolution + source.cols/2;

             output.at<unsigned char>(finger_positions[0][i-1],3*finger_positions[1][i-1]) = 255;

        }
        //Step 3.2 find nearest point on source/on depth image
        int max_distance = 20/resolution;
        int temp_distance = 20/resolution;
        int fingertip_onDepthImg[2][5] = {};

        for(int i = 0; i< 5; i++){
            if(finger_positions[2][i]!=0){
                for(int row_shift = -max_distance; row_shift <= max_distance; row_shift++){
                    for(int col_shift = -max_distance; col_shift <= max_distance; col_shift++){
                        if(row_shift*row_shift+col_shift*col_shift <= max_distance*max_distance
                                &&
                                (source.at<unsigned char>(finger_positions[0][i]+row_shift, finger_positions[1][i]+col_shift)-finger_positions[2][i])*(source.at<unsigned char>(finger_positions[0][i]+row_shift, finger_positions[1][i]+col_shift)-finger_positions[2][i])+
                                row_shift*row_shift+col_shift*col_shift <= temp_distance*temp_distance
                                &&
                                source.at<unsigned char>(finger_positions[0][i]+row_shift, finger_positions[1][i]+col_shift) != 0){
                            temp_distance = (source.at<unsigned char>(finger_positions[0][i]+row_shift, finger_positions[1][i]+col_shift)-finger_positions[2][i])*(source.at<unsigned char>(finger_positions[0][i]+row_shift, finger_positions[1][i]+col_shift)-finger_positions[2][i])+
                                    row_shift*row_shift+col_shift*col_shift;
                            temp_row = finger_positions[0][i]+row_shift;
                            temp_col = finger_positions[1][i]+col_shift;
                        }
                    }
                }
                fingertip_onDepthImg[0][i] = temp_row;
                fingertip_onDepthImg[1][i] = temp_col;
                output.at<unsigned char>(temp_row, temp_col*3+1) = 255;
                //
                markers.at<int>(temp_row,temp_col) = i*50+50;
                temp_distance = 20/resolution;
            }
        }

        //Step 3.3 watershed to find fingers

        Mat binary_Img;
        binary_Img = Mat::zeros(source.rows,source.cols,CV_8UC1);
        threshold(source, binary_Img, 1, 255, THRESH_BINARY);

        Mat dilate_binary_Img;
        dilate(binary_Img,dilate_binary_Img, Mat(), cv::Point(-1,-1), 2);
        dilate_binary_Img = 255-dilate_binary_Img;

        markers.convertTo(markers, CV_8UC1);
        markers += dilate_binary_Img;
        imshow("markers", markers);
        markers.convertTo(markers, CV_32SC1);

        Mat source_3C;
        source_3C = Mat::zeros(source.rows,source.cols,CV_8UC3);
        cvtColor(source, source_3C, COLOR_GRAY2BGR);
        //imshow("source_3C", source_3C);

        //imshow("Binary", binary_Img);
        //imshow("Eroded", eroded_binary_Img);
        watershed(source_3C, markers);
        markers.convertTo(markers, CV_8UC1);
        //imshow("water", markers);
        //waitKey();

        //Step 4 Color the Segmentation Image
        for(int row = 0; row < markers.rows; row++){
            for(int col = 0; col < markers.cols; col++){
                if(markers.at<unsigned char>(row, col) != 255 ){
                    if( markers.at<unsigned char>(row, col) != 0){
                        int i = markers.at<unsigned char> (row, col) / 50 -1;
                        int r = i * 60;
                        int b = 255- i *60;
                        output.at<unsigned char>(row, col*3) = b;
                        output.at<unsigned char>(row, col*3+2) = r;
                    }
                    else{
                        output.at<unsigned char>(row, col*3) = 255;
                        output.at<unsigned char>(row, col*3+1) = 255;
                        output.at<unsigned char>(row, col*3+2) = 255;
                    }
                }
            }
        }


    }
};
#endif
