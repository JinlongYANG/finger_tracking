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
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/TransformStamped.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
//#include <opencv2/core/eigen.hpp>
//#include <opencv2/contrib/contrib.hpp>
#include <image_geometry/pinhole_camera_model.h>
//#include <image_geometry/stereo_camera_model.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
//#include <tf2_ros/transform_broadcaster.h>
//#include <tf2_ros/transform_listener.h>
//#include "finger_tracking/finger_tracking_Config.h"
#include "finger_tracking/finger_tracking.hpp"
#include <leap_msgs/Leap.h>
#include <cstdio>

#include "GCoptimization/GCoptimization.h"



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

    message_filters::TimeSynchronizer<PointCloud2, PointCloud2, Image> timeSynchronizer_;
    message_filters::Subscriber<PointCloud2> hand_kp_Subscriber_;
    message_filters::Subscriber<PointCloud2> hand_Subscriber_;
    message_filters::Subscriber<Image> registered_Depth_Image_Subscriber;

    //    dynamic_reconfigure::Server<finger_tracking::finger_tracking_Config> reconfigureServer_;
    //    dynamic_reconfigure::Server<finger_tracking::finger_tracking_Config>::CallbackType reconfigureCallback_;

public:
    //    tf2_ros::TransformBroadcaster transformBroadcaster_;
    //    tf2_ros::Buffer buffer_;
    //    tf2_ros::TransformListener transformListener_;
    
    ros::Publisher articulatePublisher_;
    ros::Publisher handPublisher_;
    ros::Publisher necloudPublisher_;
    ros::Publisher bone_pub_;
    boost::shared_ptr<Finger_tracking> finger_tracking_;


    Finger_tracking_Node(ros::NodeHandle& nh);
    //void updateConfig(finger_tracking::finger_tracking_Config &config, uint32_t level);

    /////////////////////////////////////////////////////////////////////
    //***************     Call back function     **********************//
    void syncedCallback(const PointCloud2ConstPtr& hand_kp_pter, const PointCloud2ConstPtr& hand_pter, const ImageConstPtr& cvpointer_depthImage);
    /////////////////     end of call back function     /////////////////

    /////////////////////////////////////////////////////////////////////
    //*************      Watershed segmentation   *********************//
    void segment(Mat source, Mat & output, pcl::PointCloud<pcl::PointXYZRGB> hand_kp, int resolution, Mat & markers){
        int palmcenter_row = source.rows/2;
        int palmcenter_col = palmcenter_row;
        int palm_radius = 45/resolution;
        std::cout<<"palm_radius: "<<palm_radius<<std::endl;
        int count = 0;
        float temp_row = 0, temp_col = 0;
        //Step 1: locate center of palm(initialization is in the center of image)
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
        int fingertip_positions[3][5] = {};
        for(int i = 1; i <= 5; i++){
            //row
            fingertip_positions[0][i-1] = int(hand_kp.points[i*6-2].y * 1000)/resolution + source.rows/2;
            //col
            fingertip_positions[1][i-1] = int(hand_kp.points[i*6-2].x * 1000)/resolution + source.cols/2;
            //value
            fingertip_positions[2][i-1] = int(hand_kp.points[i*6-2].z * 1000)/resolution + source.cols/2;

            output.at<unsigned char>(fingertip_positions[0][i-1],3*fingertip_positions[1][i-1]) = 255;

        }
        //Step 3.1 read and locate fingerend positions
        int fingerend_positions[3][5] = {};
        for(int i = 1; i <= 5; i++){
            //row
            fingerend_positions[0][i-1] = int(hand_kp.points[i*6-2].y * 1000)/resolution + source.rows/2;
            //col
            fingerend_positions[1][i-1] = int(hand_kp.points[i*6-2].x * 1000)/resolution + source.cols/2;
            //value
            fingerend_positions[2][i-1] = int(hand_kp.points[i*6-2].z * 1000)/resolution + source.cols/2;

            output.at<unsigned char>(fingerend_positions[0][i-1],3*fingerend_positions[1][i-1]) = 255;

        }
        //Step 3.2 find nearest point on source/on depth image
        int max_distance = 40/resolution;
        int temp_distance_square = max_distance*max_distance;
        int fingertip_onDepthImg[2][5] = {};

        Mat binary_Img;
        binary_Img = Mat::zeros(source.rows,source.cols,CV_8UC1);
        threshold(source, binary_Img, 1, 255, THRESH_BINARY);

        Mat erode_binary_Img;
        erode(binary_Img,erode_binary_Img, Mat(), cv::Point(-1,-1), 2);
        erode_binary_Img = erode_binary_Img/255;

        Mat erode_source = erode_binary_Img.mul(source);
        //imshow("erode_source", erode_source);
        //waitKey();
        bool flag = false;

        for(int i = 0; i< 5; i++){
            //finger tip
            if(fingertip_positions[2][i]!=0){
                for(int row_shift = -max_distance; row_shift <= max_distance; row_shift++){
                    for(int col_shift = -max_distance; col_shift <= max_distance; col_shift++){
                        if(row_shift*row_shift+col_shift*col_shift <= max_distance*max_distance
                                &&
                                (erode_source.at<unsigned char>(fingertip_positions[0][i]+row_shift, fingertip_positions[1][i]+col_shift)-fingertip_positions[2][i])*(erode_source.at<unsigned char>(fingertip_positions[0][i]+row_shift, fingertip_positions[1][i]+col_shift)-fingertip_positions[2][i])+
                                row_shift*row_shift+col_shift*col_shift <= temp_distance_square
                                &&
                                erode_source.at<unsigned char>(fingertip_positions[0][i]+row_shift, fingertip_positions[1][i]+col_shift) != 0){
                            temp_distance_square = (erode_source.at<unsigned char>(fingertip_positions[0][i]+row_shift, fingertip_positions[1][i]+col_shift)-fingertip_positions[2][i])*(erode_source.at<unsigned char>(fingertip_positions[0][i]+row_shift, fingertip_positions[1][i]+col_shift)-fingertip_positions[2][i])+
                                    row_shift*row_shift+col_shift*col_shift;
                            temp_row = fingertip_positions[0][i]+row_shift;
                            temp_col = fingertip_positions[1][i]+col_shift;
                            flag = true;
                        }
                    }
                }
                if(flag == true){
                    fingertip_onDepthImg[0][i] = temp_row;
                    fingertip_onDepthImg[1][i] = temp_col;
                    output.at<unsigned char>(temp_row, temp_col*3+1) = 255;
                    //
                    markers.at<int>(temp_row,temp_col) = i*50+50;
                    flag = false;
                }
                temp_distance_square = max_distance*max_distance;
            }
            //finger end
            for(int row_shift = -max_distance; row_shift <= max_distance; row_shift++){
                for(int col_shift = -max_distance; col_shift <= max_distance; col_shift++){
                    if(row_shift*row_shift+col_shift*col_shift <= max_distance*max_distance
                            &&
                            (erode_source.at<unsigned char>(fingerend_positions[0][i]+row_shift, fingerend_positions[1][i]+col_shift)-fingerend_positions[2][i])*(erode_source.at<unsigned char>(fingerend_positions[0][i]+row_shift, fingerend_positions[1][i]+col_shift)-fingerend_positions[2][i])+
                            row_shift*row_shift+col_shift*col_shift <= temp_distance_square
                            &&
                            erode_source.at<unsigned char>(fingerend_positions[0][i]+row_shift, fingerend_positions[1][i]+col_shift) != 0){
                        temp_distance_square = (erode_source.at<unsigned char>(fingerend_positions[0][i]+row_shift, fingerend_positions[1][i]+col_shift)-fingerend_positions[2][i])*(erode_source.at<unsigned char>(fingerend_positions[0][i]+row_shift, fingerend_positions[1][i]+col_shift)-fingerend_positions[2][i])+
                                row_shift*row_shift+col_shift*col_shift;
                        temp_row = fingerend_positions[0][i]+row_shift;
                        temp_col = fingerend_positions[1][i]+col_shift;
                        flag = true;
                    }
                }
            }
            if(flag == true){
                output.at<unsigned char>(temp_row, temp_col*3+1) = 255;
                //
                markers.at<int>(temp_row,temp_col) = i*50+50;
                flag = false;
            }
            temp_distance_square = max_distance*max_distance;
        }

        //Step 3.3 watershed to find fingers



        Mat dilate_binary_Img;
        dilate(binary_Img,dilate_binary_Img, Mat(), cv::Point(-1,-1), 2);
        dilate_binary_Img = 255-dilate_binary_Img;

        markers.convertTo(markers, CV_8UC1);
        markers += dilate_binary_Img;
        //imshow("markers", markers);
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
    ////////////// end of Watershed Segmentation ////////////////////////


    /////////////////////////////////////////////////////////////////////
    //*******     Multi-label graph cut segmentation   ****************//
    void GridGraph(Mat Hand_depth, pcl::PointCloud<pcl::PointXYZRGB> Hand_kp_2d, Mat & output){
        // in this version, set data and smoothness terms using arrays
        // grid neighborhood is set up "manually". Uses spatially varying terms. Namely
        // V(p1,p2,l1,l2) = w_{p1,p2}*[min((l1-l2)*(l1-l2),4)], with
        // w_{p1,p2} = p1+p2 if |p1-p2| == 1 and w_{p1,p2} = p1*p2 if |p1-p2| is not 1

        int m_width = Hand_depth.cols;
        int m_height = Hand_depth.rows;
        int m_num_pixels = m_width * m_height;
        int m_num_labels = 17;
        int *m_result = new int[m_num_pixels];   // stores result of optimization

        Mat L_distance;
        L_distance = Mat::zeros(m_height,m_width,CV_8U);

        //prepare the joints data: 0~7 are 8 joints on palm; 8~11 are 4 joints of thumb;
        //12~15 are 4 joints of index finger; 16~19 middle finger; 20~23 ring finger; 24~27 pinky finger
        pcl::PointCloud<pcl::PointXYZRGB> Hand_joints;
        //palm: (8 9 14 15 20 21 26 27)
        for(int i = 8; i<27;i=i+6){
            Hand_joints.push_back(Hand_kp_2d.points[i]);
            Hand_joints.push_back(Hand_kp_2d.points[i+1]);
        }
        //fingers: thumb 3,4,5,6; index 9,10,11,12; middle 15,16,17,18; ring 21,22,23,24; pinky 27,28,29,30
        for(int i = 3; i<28; i = i+6){
            for(int j = 0; j<4; j++){
                Hand_joints.push_back(Hand_kp_2d.points[i+j]);
            }
        }

        // first set up the array for data costs
        int *m_data = new int[m_num_pixels*m_num_labels];
        for ( int row = 0; row < Hand_depth.rows; row++){
            for ( int col = 0; col < Hand_depth.cols; col++){
                // if the pixel is not hand
                if( Hand_depth.at<unsigned char>(row, col) == 0){
                    m_data[(col + row * m_width) * m_num_labels] = 0;
                    for (int l = 1; l < m_num_labels; l++ ){
                        m_data[(col + row * m_width) * m_num_labels + l] = 255;
                    }
                }
                //if the pixel is hand point
                else {
                    //l==0: not hand
                    m_data[(col + row * m_width) * m_num_labels] = 255;
                    //l==1: palm
                    float a = 0, b = 0, c = 0;
                    //index finger matacarpal; middle finger matacarpal; ring finger matacarpal; pinky finger matacarpal
                    float temp_min = 255;
                    for ( int f = 0; f < 4; f++){
                        int pre = 2*f;
                        int next = pre + 1;
                        a = sqrt((Hand_joints.points[pre].x - Hand_joints.points[next].x)*(Hand_joints.points[pre].x - Hand_joints.points[next].x)
                                 +(Hand_joints.points[pre].y - Hand_joints.points[next].y) * (Hand_joints.points[pre].y - Hand_joints.points[next].y)
                                 +(Hand_joints.points[pre].z - Hand_joints.points[next].z) * (Hand_joints.points[pre].z - Hand_joints.points[next].z));
                        b = sqrt((Hand_joints.points[pre].x - row)*(Hand_joints.points[pre].x - row)
                                 + (Hand_joints.points[pre].y - col)*(Hand_joints.points[pre].y - col)
                                 + (Hand_joints.points[pre].z - Hand_depth.at<unsigned char>(row, col))*(Hand_joints.points[pre].z - Hand_depth.at<unsigned char>(row, col)));
                        c = sqrt((Hand_joints.points[next].x - row)*(Hand_joints.points[next].x - row)
                                 + (Hand_joints.points[next].y - col)*(Hand_joints.points[next].y - col)
                                 + (Hand_joints.points[next].z - Hand_depth.at<unsigned char>(row, col))*(Hand_joints.points[next].z - Hand_depth.at<unsigned char>(row, col)));
                        //if point projectiong is out of link; else on link
                        if( a*a + min(b,c)*min(b,c)< max(b,c)*max(b,c)){
                            if(temp_min > min(b,c))
                                temp_min = min(b,c);
                        }
                        else{
                            if ( temp_min > 0.5*sqrt((a+b+c)*(a+b-c)*(b+c-a)*(a-b+c))/a )
                                temp_min = 0.5*sqrt((a+b+c)*(a+b-c)*(b+c-a)*(a-b+c))/a;
                        }
                    }
                    m_data[(col + row * m_width) * m_num_labels + 1] = temp_min;


                    //l==2~17: fingers
                    //l==2: thumb metacarpal; l==3: thumb proximal; l==4: thumb distal
                    //l==5: index finger proximal; l==6: index finger intermediate; l==7: index finger distal
                    //l==8: middle finger proximal; l==9: middle finger intermediate; l==10: middle finger distal
                    //l==11: ring finger proximal; l==12: index ring intermediate; l==13: ring finger distal
                    //l==14: pinky finger proximal; l==15: pinky finger intermediate; l==16: pinky finger distal
                    for ( int f = 0; f < 5; f++){
                        for ( int k = 0; k< 3; k++){
                            int pre = 4*f+k+8;
                            int next = pre+1;
                            int l = f*3+k+2;
                            a = sqrt((Hand_joints.points[pre].x - Hand_joints.points[next].x)*(Hand_joints.points[pre].x - Hand_joints.points[next].x)
                                     +(Hand_joints.points[pre].y - Hand_joints.points[next].y) * (Hand_joints.points[pre].y - Hand_joints.points[next].y)
                                     +(Hand_joints.points[pre].z - Hand_joints.points[next].z) * (Hand_joints.points[pre].z - Hand_joints.points[next].z));
                            b = sqrt((Hand_joints.points[pre].x - row)*(Hand_joints.points[pre].x - row)
                                     + (Hand_joints.points[pre].y - col)*(Hand_joints.points[pre].y - col)
                                     + (Hand_joints.points[pre].z - Hand_depth.at<unsigned char>(row, col))*(Hand_joints.points[pre].z - Hand_depth.at<unsigned char>(row, col)));
                            c = sqrt((Hand_joints.points[next].x - row)*(Hand_joints.points[next].x - row)
                                     + (Hand_joints.points[next].y - col)*(Hand_joints.points[next].y - col)
                                     + (Hand_joints.points[next].z - Hand_depth.at<unsigned char>(row, col))*(Hand_joints.points[next].z - Hand_depth.at<unsigned char>(row, col)));
                            //if point projectiong is out of link; else on link
                            if( a*a + min(b,c)*min(b,c)< max(b,c)*max(b,c)){
                                m_data[(col + row * m_width) * m_num_labels + l] = min(b,c);
                            }
                            else{
                                m_data[(col + row * m_width) * m_num_labels + l] = 0.5*sqrt((a+b+c)*(a+b-c)*(b+c-a)*(a-b+c))/a;
                            }
                        }
                    }
                }
                L_distance.at<unsigned char>(row, col) = min(int(10*m_data[(col + row * m_width) * m_num_labels + 9]), 255);
            }
        }
        imshow("L_distance", L_distance);


        // next set up the array for smooth costs
        int *smooth = new int[m_num_labels*m_num_labels];
        for ( int l1 = 0; l1 < m_num_labels; l1++ )
            for (int l2 = 0; l2 < m_num_labels; l2++ )
                smooth[l1+l2*m_num_labels] = 100;


        try{
            GCoptimizationGeneralGraph *gc = new GCoptimizationGeneralGraph(m_num_pixels,m_num_labels);


            gc->setDataCost(m_data);
            gc->setSmoothCost(smooth);

            // now set up a grid neighborhood system
            // first set up horizontal neighbors
            for (int y = 0; y < m_height; y++ )
                for (int  x = 1; x < m_width; x++ )
                    if (abs(Hand_depth.at<unsigned char>(y,x)-Hand_depth.at<unsigned char>(y, x-1))<4)
                        gc->setNeighbors(x+y*m_width,x-1+y*m_width);

            // next set up vertical neighbors
            for (int y = 1; y < m_height; y++ )
                for (int  x = 0; x < m_width; x++ )
                    if (abs(Hand_depth.at<unsigned char>(y,x)-Hand_depth.at<unsigned char>(y-1, x))<4)
                        gc->setNeighbors(x+y*m_width,x+(y-1)*m_width);

            printf("Before optimization energy is %lld\n",gc->compute_energy());
            //            for ( int  i = 0; i < m_num_pixels; i++ ){
            //                m_result[i] = gc->whatLabel(i);
            //                std::cout<<m_result[i]<<" ";
            //                if((i+1)%10 == 0)
            //                    std::cout<<std::endl;
            //            }
            gc->expansion(15);// run expansion for 1 iterations. For swap use gc->swap(num_iterations);
            gc->swap(3);
            printf("After optimization energy is %lld\n",gc->compute_energy());

            //            for ( int  i = 0; i < m_num_pixels; i++ ){
            //                m_result[i] = gc->whatLabel(i);
            //                std::cout<<m_result[i]<<" ";
            //                if((i+1)%10 == 0)
            //                    std::cout<<std::endl;
            //            }


            for ( int row = 0; row < Hand_depth.rows; row++){
                for ( int col = 0; col < Hand_depth.cols; col++){
                    int label = gc->whatLabel(row * Hand_depth.cols + col);
                    if(label == 0){
                        output.at<unsigned char>(row, 3*col+0) = 255;
                        output.at<unsigned char>(row, 3*col+1) = 255;
                        output.at<unsigned char>(row, 3*col+2) = 255;
                    }
                    else if(label%3 == 0)
                        output.at<unsigned char>(row, 3*col+0) = label*15;
                    if(label%3 == 1)
                        output.at<unsigned char>(row, 3*col+1) = label*15;
                    if(label%3 == 2)
                        output.at<unsigned char>(row, 3*col+2) = label*15;

                }
            }

            delete gc;
        }
        catch (GCException e){
            e.Report();
        }

        delete [] m_result;
        delete [] smooth;
        delete [] m_data;
        //cv::waitKey();


    }

    ////////////// end of Multi-label graph cut Segmentation ////////////

    /////////////////////////////////////////////////////////////////////
    //*******     Multi-label graph cut segmentation   ****************//
    void GridGraphSeqeratePalm(Mat Hand_depth, pcl::PointCloud<pcl::PointXYZRGB> Hand_kp_2d, Mat & output){
        // in this version, set data and smoothness terms using arrays
        // grid neighborhood is set up "manually". Uses spatially varying terms. Namely
        // V(p1,p2,l1,l2) = w_{p1,p2}*[min((l1-l2)*(l1-l2),4)], with
        // w_{p1,p2} = p1+p2 if |p1-p2| == 1 and w_{p1,p2} = p1*p2 if |p1-p2| is not 1

        int m_width = Hand_depth.cols;
        int m_height = Hand_depth.rows;
        int m_num_pixels = m_width * m_height;
        int m_num_labels = 20;
        int *m_result = new int[m_num_pixels];   // stores result of optimization

        Mat L_distance;
        L_distance = Mat::zeros(m_height,m_width,CV_8U);

        //prepare the joints data: 0~7 are 8 joints on palm; 8~11 are 4 joints of thumb;
        //12~15 are 4 joints of index finger; 16~19 middle finger; 20~23 ring finger; 24~27 pinky finger
        pcl::PointCloud<pcl::PointXYZRGB> Hand_joints;
        //palm: (8 9 14 15 20 21 26 27)
        for(int i = 8; i<27;i=i+6){
            Hand_joints.push_back(Hand_kp_2d.points[i]);
            Hand_joints.push_back(Hand_kp_2d.points[i+1]);
        }
        //fingers: thumb 3,4,5,6; index 9,10,11,12; middle 15,16,17,18; ring 21,22,23,24; pinky 27,28,29,30
        for(int i = 3; i<28; i = i+6){
            for(int j = 0; j<4; j++){
                Hand_joints.push_back(Hand_kp_2d.points[i+j]);
            }
        }

        // first set up the array for data costs
        int *m_data = new int[m_num_pixels*m_num_labels];
        for ( int row = 0; row < Hand_depth.rows; row++){
            for ( int col = 0; col < Hand_depth.cols; col++){
                // if the pixel is not hand
                if( Hand_depth.at<unsigned char>(row, col) == 0){
                    m_data[(col + row * m_width) * m_num_labels] = 0;
                    for (int l = 1; l < m_num_labels; l++ ){
                        m_data[(col + row * m_width) * m_num_labels + l] = 255;
                    }
                }
                //if the pixel is hand point
                else {
                    //l==0: not hand
                    m_data[(col + row * m_width) * m_num_labels] = 255;
                    //l==1~4: palm
                    float a = 0, b = 0, c = 0;
                    //index finger matacarpal; middle finger matacarpal; ring finger matacarpal; pinky finger matacarpal
                    for ( int f = 0; f < 4; f++){
                        int pre = 2*f;
                        int next = pre + 1;
                        a = sqrt((Hand_joints.points[pre].x - Hand_joints.points[next].x)*(Hand_joints.points[pre].x - Hand_joints.points[next].x)
                                 +(Hand_joints.points[pre].y - Hand_joints.points[next].y) * (Hand_joints.points[pre].y - Hand_joints.points[next].y)
                                 +(Hand_joints.points[pre].z - Hand_joints.points[next].z) * (Hand_joints.points[pre].z - Hand_joints.points[next].z));
                        b = sqrt((Hand_joints.points[pre].x - row)*(Hand_joints.points[pre].x - row)
                                 + (Hand_joints.points[pre].y - col)*(Hand_joints.points[pre].y - col)
                                 + (Hand_joints.points[pre].z - Hand_depth.at<unsigned char>(row, col))*(Hand_joints.points[pre].z - Hand_depth.at<unsigned char>(row, col)));
                        c = sqrt((Hand_joints.points[next].x - row)*(Hand_joints.points[next].x - row)
                                 + (Hand_joints.points[next].y - col)*(Hand_joints.points[next].y - col)
                                 + (Hand_joints.points[next].z - Hand_depth.at<unsigned char>(row, col))*(Hand_joints.points[next].z - Hand_depth.at<unsigned char>(row, col)));
                        //if point projectiong is out of link; else on link
                        if( a*a + min(b,c)*min(b,c)< max(b,c)*max(b,c)){
                                m_data[(col + row * m_width) * m_num_labels + 1 + f] = min(b,c);
                        }
                        else{
                                m_data[(col + row * m_width) * m_num_labels + 1 + f] = 0.5*sqrt((a+b+c)*(a+b-c)*(b+c-a)*(a-b+c))/a;
                        }
                    }


                    //l==5~19: fingers
                    //l==5: thumb metacarpal; l==6: thumb proximal; l==7: thumb distal
                    //l==8: index finger proximal; l==9: index finger intermediate; l==10: index finger distal
                    //l==11: middle finger proximal; l==12: middle finger intermediate; l==13: middle finger distal
                    //l==14: ring finger proximal; l==15: index ring intermediate; l==16: ring finger distal
                    //l==17: pinky finger proximal; l==18: pinky finger intermediate; l==19: pinky finger distal
                    for ( int f = 0; f < 5; f++){
                        for ( int k = 0; k< 3; k++){
                            int pre = 4*f+k+8;
                            int next = pre+1;
                            int l = f*3+k+5;
                            a = sqrt((Hand_joints.points[pre].x - Hand_joints.points[next].x)*(Hand_joints.points[pre].x - Hand_joints.points[next].x)
                                     +(Hand_joints.points[pre].y - Hand_joints.points[next].y) * (Hand_joints.points[pre].y - Hand_joints.points[next].y)
                                     +(Hand_joints.points[pre].z - Hand_joints.points[next].z) * (Hand_joints.points[pre].z - Hand_joints.points[next].z));
                            b = sqrt((Hand_joints.points[pre].x - row)*(Hand_joints.points[pre].x - row)
                                     + (Hand_joints.points[pre].y - col)*(Hand_joints.points[pre].y - col)
                                     + (Hand_joints.points[pre].z - Hand_depth.at<unsigned char>(row, col))*(Hand_joints.points[pre].z - Hand_depth.at<unsigned char>(row, col)));
                            c = sqrt((Hand_joints.points[next].x - row)*(Hand_joints.points[next].x - row)
                                     + (Hand_joints.points[next].y - col)*(Hand_joints.points[next].y - col)
                                     + (Hand_joints.points[next].z - Hand_depth.at<unsigned char>(row, col))*(Hand_joints.points[next].z - Hand_depth.at<unsigned char>(row, col)));
                            //if point projectiong is out of link; else on link
                            if( a*a + min(b,c)*min(b,c)< max(b,c)*max(b,c)){
                                m_data[(col + row * m_width) * m_num_labels + l] = min(b,c);
                            }
                            else{
                                m_data[(col + row * m_width) * m_num_labels + l] = 0.5*sqrt((a+b+c)*(a+b-c)*(b+c-a)*(a-b+c))/a;
                            }
                        }
                    }
                }
                L_distance.at<unsigned char>(row, col) = min(int(10*m_data[(col + row * m_width) * m_num_labels + 9]), 255);
            }
        }
        //imshow("L_distance", L_distance);


        // next set up the array for smooth costs
        int *smooth = new int[m_num_labels*m_num_labels];
        for ( int l1 = 0; l1 < m_num_labels; l1++ )
            for (int l2 = 0; l2 < m_num_labels; l2++ )
                smooth[l1+l2*m_num_labels] = 100;


        try{
            GCoptimizationGeneralGraph *gc = new GCoptimizationGeneralGraph(m_num_pixels,m_num_labels);


            gc->setDataCost(m_data);
            gc->setSmoothCost(smooth);

            // now set up a grid neighborhood system
            // first set up horizontal neighbors
            for (int y = 0; y < m_height; y++ )
                for (int  x = 1; x < m_width; x++ )
                    if (abs(Hand_depth.at<unsigned char>(y,x)-Hand_depth.at<unsigned char>(y, x-1))<4)
                        gc->setNeighbors(x+y*m_width,x-1+y*m_width);

            // next set up vertical neighbors
            for (int y = 1; y < m_height; y++ )
                for (int  x = 0; x < m_width; x++ )
                    if (abs(Hand_depth.at<unsigned char>(y,x)-Hand_depth.at<unsigned char>(y-1, x))<4)
                        gc->setNeighbors(x+y*m_width,x+(y-1)*m_width);

            //printf("Before optimization energy is %lld\n",gc->compute_energy());
            //            for ( int  i = 0; i < m_num_pixels; i++ ){
            //                m_result[i] = gc->whatLabel(i);
            //                std::cout<<m_result[i]<<" ";
            //                if((i+1)%10 == 0)
            //                    std::cout<<std::endl;
            //            }
            ros::Time time1 = ros::Time::now();
            gc->expansion(5);// run expansion for 1 iterations. For swap use gc->swap(num_iterations);
            //gc->swap(5);
            ros::Time time2 = ros::Time::now();
            std::cout<<"Time in: "<< time2-time1<<std::endl;
            //printf("After optimization energy is %lld\n",gc->compute_energy());

            //            for ( int  i = 0; i < m_num_pixels; i++ ){
            //                m_result[i] = gc->whatLabel(i);
            //                std::cout<<m_result[i]<<" ";
            //                if((i+1)%10 == 0)
            //                    std::cout<<std::endl;
            //            }


            for ( int row = 0; row < Hand_depth.rows; row++){
                for ( int col = 0; col < Hand_depth.cols; col++){
                    int label = gc->whatLabel(row * Hand_depth.cols + col);
                    if(label == 0){
                        output.at<unsigned char>(row, 3*col+0) = 255;
                        output.at<unsigned char>(row, 3*col+1) = 255;
                        output.at<unsigned char>(row, 3*col+2) = 255;
                    }
                    else if(label%3 == 0)
                        output.at<unsigned char>(row, 3*col+0) = label*13;
                    if(label%3 == 1)
                        output.at<unsigned char>(row, 3*col+1) = label*13;
                    if(label%3 == 2)
                        output.at<unsigned char>(row, 3*col+2) = label*13;

                }
            }

            delete gc;
        }
        catch (GCException e){
            e.Report();
        }

        delete [] m_result;
        delete [] smooth;
        delete [] m_data;
        //cv::waitKey();
    }

    ////////////// end of Multi-label graph cut Segmentation ////////////


};
#endif
