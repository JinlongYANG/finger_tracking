#define BOOST_SIGNALS_NO_DEPRECATION_WARNING
#include "finger_tracking/finger_tracking_node.hpp"
#include "finger_tracking/handkp_leap_msg.h"
#include "finger_tracking/Hand_XYZRGB.h"
#include "finger_tracking/articulate_HandModel_XYZRGB.h"
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
bool flag_leap = true;
pcl::PointXYZRGB my_hand_kp[31];
pcl::PointXYZRGB last_joints_position[26];





Finger_tracking_Node::Finger_tracking_Node(ros::NodeHandle& nh):
    imageTransport_(nh),
    timeSynchronizer_(10)
  //reconfigureServer_(ros::NodeHandle(nh,"finger_tracking")),
  //transformListener_(buffer_, true)
  //reconfigureCallback_(boost::bind(&finger_tracking_Node::updateConfig, this, _1, _2))
{

    hand_kp_Subscriber_.subscribe(nh, "/Hand_kp_cl", 10);
    hand_Subscriber_.subscribe(nh, "/Hand_pcl", 10);
    registered_Depth_Image_Subscriber.subscribe(nh, "/Depth_Image", 10);

    timeSynchronizer_.connectInput(hand_kp_Subscriber_, hand_Subscriber_, registered_Depth_Image_Subscriber);

    depthpublisher_ = imageTransport_.advertise("DepthImage", 0);
    segmentpublisher_ = imageTransport_.advertise("Segment", 0);

    handPublisher_ = nh.advertise<sensor_msgs::PointCloud2>("Hand",0);
    articulatePublisher_ = nh.advertise<sensor_msgs::PointCloud2>("Articulate",0);

    necloudPublisher_  = nh.advertise<sensor_msgs::PointCloud2>("Norms",0);
    bone_pub_ = nh.advertise<visualization_msgs::Marker>("Bones", 0);
    leap_articulate_pub_ = nh.advertise<visualization_msgs::Marker>("Leap_Articulate",0);

    timeSynchronizer_.registerCallback(boost::bind(&Finger_tracking_Node::syncedCallback, this, _1, _2, _3));
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

void Finger_tracking_Node::syncedCallback(const PointCloud2ConstPtr& hand_kp_pter, const PointCloud2ConstPtr& hand_pter, const ImageConstPtr& cvpointer_depthImage){

    ros::Time time0 = ros::Time::now();
    cv_bridge::CvImagePtr cvpointer_depthFrame;



    pcl::PointCloud<pcl::PointXYZRGB> msg_pcl, handcloud, hand1_kp;

    ROS_INFO("Callback begins");

    try
    {
        int seq = hand_kp_pter->header.seq;

        fromROSMsg(*hand_kp_pter, hand1_kp);
        fromROSMsg(*hand_pter, msg_pcl);

        //std::cout<<"Hand Keypoint size: "<<hand1_kp.size()<<endl;
        //std::cout<<"Hand Cloud size: "<<msg_pcl.size()<<endl;

        cvpointer_depthFrame = cv_bridge::toCvCopy(cvpointer_depthImage);
        Mat Origional_depthImage;
        Origional_depthImage = cvpointer_depthFrame->image;

        Mat Hand_DepthMat;
        Hand_DepthMat = Mat::zeros(imageSize,imageSize,CV_8U);

        for (size_t i = 0; i < msg_pcl.points.size (); ++i){
            if( (abs(msg_pcl.points[i].x) < 0.2
                 && abs(msg_pcl.points[i].y) < 0.2
                 && abs(msg_pcl.points[i].z) < 0.2)
                    &&((abs(msg_pcl.points[i].x)*abs(msg_pcl.points[i].x)+ abs(msg_pcl.points[i].y)*abs(msg_pcl.points[i].y)+abs(msg_pcl.points[i].z)*abs(msg_pcl.points[i].z)< half_palm_length*half_palm_length) ||
                       (abs(msg_pcl.points[i].x - hand1_kp.points[1].x) < finger_length
                        && abs(msg_pcl.points[i].y - hand1_kp.points[1].y) < finger_length
                        && abs(msg_pcl.points[i].z - hand1_kp.points[1].z) < finger_length)||
                       (abs(msg_pcl.points[i].x - hand1_kp.points[7].x) < finger_length
                        && abs(msg_pcl.points[i].y - hand1_kp.points[7].y) < finger_length
                        && abs(msg_pcl.points[i].z - hand1_kp.points[7].z) < finger_length)||
                       (abs(msg_pcl.points[i].x - hand1_kp.points[13].x) < finger_length
                        && abs(msg_pcl.points[i].y - hand1_kp.points[13].y) < finger_length
                        && abs(msg_pcl.points[i].z - hand1_kp.points[13].z) < finger_length)||
                       (abs(msg_pcl.points[i].x - hand1_kp.points[19].x) < finger_length
                        && abs(msg_pcl.points[i].y - hand1_kp.points[19].y) < finger_length
                        && abs(msg_pcl.points[i].z - hand1_kp.points[19].z) < finger_length)||
                       (abs(msg_pcl.points[i].x - hand1_kp.points[25].x) < finger_length
                        && abs(msg_pcl.points[i].y - hand1_kp.points[25].y) < finger_length
                        && abs(msg_pcl.points[i].z - hand1_kp.points[25].z) < finger_length))
                    ){

                //                if( (abs(msg_pcl.points[i].x) < 0.2
                //                     && abs(msg_pcl.points[i].y) < 0.2
                //                     && abs(msg_pcl.points[i].z) < 0.2)
                //                        &&((abs(msg_pcl.points[i].x)*abs(msg_pcl.points[i].x)+ abs(msg_pcl.points[i].y)*abs(msg_pcl.points[i].y)+abs(msg_pcl.points[i].z)*abs(msg_pcl.points[i].z)< half_palm_length*half_palm_length) ||
                //                           (abs(msg_pcl.points[i].x - hand1_kp.points[1].x) < finger_length
                //                            && abs(msg_pcl.points[i].y - hand1_kp.points[1].y) < finger_length
                //                            && abs(msg_pcl.points[i].z - hand1_kp.points[1].z) < finger_length)||
                //                           (abs(msg_pcl.points[i].x - hand1_kp.points[2].x) < finger_length
                //                            && abs(msg_pcl.points[i].y - hand1_kp.points[2].y) < finger_length
                //                            && abs(msg_pcl.points[i].z - hand1_kp.points[2].z) < finger_length)||
                //                           (abs(msg_pcl.points[i].x - hand1_kp.points[3].x) < finger_length
                //                            && abs(msg_pcl.points[i].y - hand1_kp.points[3].y) < finger_length
                //                            && abs(msg_pcl.points[i].z - hand1_kp.points[3].z) < finger_length)||
                //                           (abs(msg_pcl.points[i].x - hand1_kp.points[4].x) < finger_length
                //                            && abs(msg_pcl.points[i].y - hand1_kp.points[4].y) < finger_length
                //                            && abs(msg_pcl.points[i].z - hand1_kp.points[4].z) < finger_length)||
                //                           (abs(msg_pcl.points[i].x - hand1_kp.points[5].x) < finger_length
                //                            && abs(msg_pcl.points[i].y - hand1_kp.points[5].y) < finger_length
                //                            && abs(msg_pcl.points[i].z - hand1_kp.points[5].z) < finger_length))
                //                        ){

                handcloud.push_back(msg_pcl.points[i]);
                int x = int(msg_pcl.points[i].x * 1000);
                int y = int(msg_pcl.points[i].y * 1000);
                int z = int(msg_pcl.points[i].z * 1000);
                int row = y/resolution + imageSize/2;
                int col = x/resolution + imageSize/2;
                int depth = z/resolution + imageSize/2;
                //std::cout<<"row col depth:"<<row<<" "<<col<<" "<<depth<<endl;
                if( row < imageSize && col < imageSize){
                    Hand_DepthMat.at<unsigned char>(row, col) = depth;
                }

            }

        }
        //std::cout<<"Hand Cloud size: "<<handcloud.size()<<endl;
        ////////////////////////////////////////////////////////
        Mat SegmentMat, LabelMat;
        //************     Watershed segmentation   **********//
        //        Mat Hand_DepthMat4Watershed;
        //        Hand_DepthMat4Watershed = Mat::zeros(imageSize,imageSize,CV_8U);
        //        Hand_DepthMat.copyTo(Hand_DepthMat4Watershed);
        //        Mat markers= Mat::zeros(imageSize,imageSize,CV_32SC1);
        //        SegmentMat = Mat::zeros(imageSize,imageSize,CV_8UC3);
        //        segment(Hand_DepthMat4Watershed, SegmentMat, hand1_kp, resolution, markers);
        ////////////////////////////////////////////////////////

        ///////////////////////////////////////////////////////////
        //*************   Multi-label graph cut    **************//

        ros::Time time1 = ros::Time::now();

        pcl::PointCloud<pcl::PointXYZRGB> Hand_kp_2d;
        if(flag_leap == true){
            ROS_INFO("Flag leap true!");
            for(int i = 0; i < 31; i++){
                pcl::PointXYZRGB p_2d;
                int x = int(hand1_kp.points[i] . x *1000);
                int y = int(hand1_kp.points[i] . y *1000);
                int z = int(hand1_kp.points[i] . z *1000);
                p_2d.x = y/resolution + imageSize/2;
                p_2d.y = x/resolution + imageSize/2;
                p_2d.z = z/resolution + imageSize/2;
                Hand_kp_2d.push_back(p_2d);
            }
        }
        else {
            for(int i = 0; i < 31; i++){
                pcl::PointXYZRGB p_2d;
                int x = int(my_hand_kp[i] . x *1000);
                int y = int(my_hand_kp[i] . y *1000);
                int z = int(my_hand_kp[i] . z *1000);
                p_2d.x = y/resolution + imageSize/2;
                p_2d.y = x/resolution + imageSize/2;
                p_2d.z = z/resolution + imageSize/2;
                Hand_kp_2d.push_back(p_2d);
            }
        }
        //        std::cout << "Hand_kp_2d: " << std::endl;
        //        for (int i = 0; i<31; i++){
        //            std::cout << Hand_kp_2d.points[i].x <<" "<<Hand_kp_2d.points[i].y<<" "<<Hand_kp_2d.points[i].z<<std::endl;
        //        }

        SegmentMat = Mat::zeros(imageSize,imageSize,CV_8UC3);
        LabelMat = Mat::zeros(imageSize,imageSize,CV_8UC1);
        //GridGraph(Hand_DepthMat, Hand_kp_2d, SegmentMat);
        GridGraphSeqeratePalm(Hand_DepthMat, Hand_kp_2d, SegmentMat, LabelMat);
        ///////////////////////////////////////////////////////////
        ros::Time time2 = ros::Time::now();
        std::cout<<"Time: "<< time2 - time1 << std::endl;

        vector< vector<pcl::PointXYZ> > labelPointXYZ(20, vector<pcl::PointXYZ>());
        Image2Label(Hand_DepthMat, LabelMat, labelPointXYZ, resolution);

        std::cout << "Size of vector: " << labelPointXYZ.size() << std::endl;

        vector<int> labelflag(20);
        for(int i = 0; i< 20; i++){
            labelflag[i] = 1;
            if(labelPointXYZ[i].size() == 0)
                labelflag[i] = 0;
            std::cout << "Size of " << i << ": " << labelPointXYZ[i].size() << std::endl;
        }
        for(int i = 0; i < labelflag.size(); ++i){
            std::cout << "Label flag: " << labelflag[i] << std::endl;
        }

        ///////////////////////////////////////////////////////////
        //**************    Hand Model   ************************//
        pcl::PointCloud<pcl::PointXYZRGB> articulation, leap_articulation;
        articulate_HandModel_XYZRGB Hand_model, Leap_hand;

        //read in leap motion oberservation
        Leap_hand.set_joints_positions(hand1_kp);
//        Hand_model.set_joints_positions(hand1_kp);
//        Hand_model.get_parameters();
//        Hand_model.get_joints_positions();

        Hand_model.set_parameters();
        Hand_model.get_joints_positions();
        //        //Hand_model.get_parameters();
        //Palm ICP
        bool palm_fit;
        if(labelflag[1]+labelflag[2]+labelflag[3]+labelflag[4]+labelflag[5] == 5){
            for(int iteration_Number = 0; iteration_Number < 8; iteration_Number++){
                Hand_model.CP_palm_fitting3(Hand_DepthMat,LabelMat, resolution, labelflag, palm_fit);
                Hand_model.get_joints_positions();
            }
        }
        else{
            Hand_model.joints_position[1] = last_joints_position[1];
            Hand_model.joints_position[6] = last_joints_position[6];
            Hand_model.joints_position[7] = last_joints_position[7];
            Hand_model.joints_position[11] = last_joints_position[11];
            Hand_model.joints_position[12] = last_joints_position[12];
            Hand_model.joints_position[16] = last_joints_position[16];
            Hand_model.joints_position[17] = last_joints_position[17];
            Hand_model.joints_position[21] = last_joints_position[21];
            Hand_model.joints_position[22] = last_joints_position[22];
        }

        //fitting1: average vector direction


        //     //////////////////   Finger fitting 2 ///////////////////////////
        //                      //fitting2: ICP
        //        for(int ite = 0; ite < 2; ite++){
        //            //Proximal fitting:
        //            for(int i = 0; i < 2; i++){
        //                Hand_model.finger_fitting2(Hand_DepthMat,LabelMat, resolution, 0);
        //            }
        //            //Intermediate fitting:
        //            for(int i = 0; i < 2; i++){
        //                Hand_model.finger_fitting2(Hand_DepthMat,LabelMat, resolution, 1);
        //                Hand_model.constrain_based_smooth( 2 );
        //            }
        //            //distal fitting:
        //            for(int i = 0; i < 2; i++){
        //                Hand_model.finger_fitting2(Hand_DepthMat,LabelMat, resolution, 2);
        //                Hand_model.constrain_based_smooth( 3 );
        //            }

        //            //constrain based smooth

        //        }
        //                Hand_model.get_parameters();
        //                Hand_model.get_joints_positions();
        //      //////////////    End of finger fitting 2   /////////////////////


        //        //        //////////////////   Finger fitting 3 ///////////////////////////
        //        //                        fitting3: Ransac line fitting
        //                Hand_model.finger_fitting3(labelPointXYZ, 0);
        //                Hand_model.finger_fitting3(labelPointXYZ, 1);
        //                Hand_model.finger_fitting3(labelPointXYZ, 2);
        //                Hand_model.constrain_based_smooth( 3 );

        //        //        //////////////    End of finger fitting 3   /////////////////////

//                //////////////////   Finger fitting 4 ///////////////////////////
//                        //    fitting4: Searching for intersection
//                Hand_model.finger_fitting4(Hand_DepthMat,LabelMat, resolution, 0);
//                //Hand_model.constrain_based_smooth( 2 );
//                Hand_model.finger_fitting3(labelPointXYZ, 2);
//                //Hand_model.constrain_based_smooth( 3 );

//        ////////////    End of finger fitting 4   /////////////////////

//        ////////////////   Finger fitting 5 ///////////////////////////
//            //fitting5: Intersection area ransac
//        Hand_model.finger_fitting5(Hand_DepthMat,LabelMat, resolution, 15, labelPointXYZ);
//        Hand_model.constrain_based_smooth( 3 );

//        ////////////    End of finger fitting 5   /////////////////////

                ////////////////   Finger fitting 6 ///////////////////////////
                    //fitting5: Intersection area ransac
        if(labelflag[5]+labelflag[6]+labelflag[7] == 3)
                Hand_model.finger_fitting6(Hand_DepthMat,LabelMat, resolution, 15, labelPointXYZ, 1);
        else{
            ;
        }
        if(labelflag[8]+labelflag[9]+labelflag[10] == 3)
                Hand_model.finger_fitting6(Hand_DepthMat,LabelMat, resolution, 15, labelPointXYZ, 2);
        else{
            ;
        }


        if(labelflag[11]+labelflag[12]+labelflag[13] == 3)
                Hand_model.finger_fitting6(Hand_DepthMat,LabelMat, resolution, 15, labelPointXYZ, 3);
        else{
            ;
        }


        if(labelflag[14]+labelflag[15]+labelflag[16] == 3)
                Hand_model.finger_fitting6(Hand_DepthMat,LabelMat, resolution, 15, labelPointXYZ, 4);
        else{
            ;
        }


        if(labelflag[17]+labelflag[18]+labelflag[19] == 3)
                Hand_model.finger_fitting6(Hand_DepthMat,LabelMat, resolution, 15, labelPointXYZ, 5);
        else{
            ;
        }


                Hand_model.constrain_based_smooth( 3 );

                ////////////    End of finger fitting 6   /////////////////////

        //        //
        //        //Hand_model.finger_fitting2_proximal(Hand_DepthMat,LabelMat, resolution);

        //        Hand_model.trial();



        //        pcl::PointXYZRGB raw_joints_positions[26];
        //        for(int i = 0; i < 26; i++){
        //            raw_joints_positions[i] = Hand_model.joints_position[i];
        //        }
        //        for(int iter = 0; iter < 5; iter++){
        //            Hand_model.get_parameters();
        //            Hand_model.get_joints_positions();
        //            for(int i = 0; i < 26; i++){
        //                Hand_model.joints_position[i].x = 0.9*raw_joints_positions[i].x + 0.1*Hand_model.joints_position[i].x;
        //                Hand_model.joints_position[i].y = 0.9*raw_joints_positions[i].y + 0.1*Hand_model.joints_position[i].y;
        //                Hand_model.joints_position[i].z = 0.9*raw_joints_positions[i].z + 0.1*Hand_model.joints_position[i].z;
        //            }
        //        }

        //        Hand_model.get_parameters();
        //        Hand_model.get_joints_positions();




        for(int i = 0; i < 26; i++){
            articulation.push_back(Hand_model.joints_position[i]);
            SegmentMat.at<unsigned char>(int(Hand_model.joints_position[i].y*1000)/resolution + imageSize/2,
                                         3*(int(Hand_model.joints_position[i].x*1000)/resolution + imageSize/2)+2) = 180;
            SegmentMat.at<unsigned char>(int(Hand_model.joints_position[i].y*1000)/resolution + imageSize/2,
                                         3*(int(Hand_model.joints_position[i].x*1000)/resolution + imageSize/2)+1) = 180;
            SegmentMat.at<unsigned char>(int(Hand_model.joints_position[i].y*1000)/resolution + imageSize/2,
                                         3*(int(Hand_model.joints_position[i].x*1000)/resolution + imageSize/2)+0) = 0;
            leap_articulation.push_back(Leap_hand.joints_position[i]);
            last_joints_position[i] = Hand_model.joints_position[i];
        }



        //////////// prepare for next frame  //////////////////////

        my_hand_kp[0] = hand1_kp.points[0];
        for(int i = 0 ; i < 5; i++){
            //fingertips:
            my_hand_kp[6*i+1] = Hand_model.joints_position[5+i*5];
            //joints:
            for(int j = 0; j< 5; j++){
                my_hand_kp[6*i+j+2] = Hand_model.joints_position[1+j+i*5];
            }
        }


        //flag_leap = false;




        ///////////////////////////////////////////////////////////

        ROS_INFO("Prepare Hand Cloud");
        sensor_msgs::PointCloud2 cloud_msg;
        toROSMsg(handcloud,cloud_msg);
        cloud_msg.header.frame_id=hand_kp_pter->header.frame_id;
        cloud_msg.header.stamp = hand_kp_pter->header.stamp;
        handPublisher_.publish(cloud_msg);
        ROS_INFO("Prepare Articulation");
        sensor_msgs::PointCloud2 cloud_msg_articulation;
        toROSMsg(articulation,cloud_msg_articulation);
        cloud_msg_articulation.header.frame_id=hand_kp_pter->header.frame_id;
        cloud_msg_articulation.header.stamp = hand_kp_pter->header.stamp;
        articulatePublisher_.publish(cloud_msg_articulation);
        ROS_INFO("Prepare Depth Image");
        cv_bridge::CvImage depthImage_msg;
        depthImage_msg.encoding = sensor_msgs::image_encodings::MONO8;
        depthImage_msg.image    = Hand_DepthMat;
        depthImage_msg.header.seq = seq;
        depthImage_msg.header.frame_id = hand_kp_pter->header.frame_id;;
        depthImage_msg.header.stamp =  hand_kp_pter->header.stamp;
        depthpublisher_.publish(depthImage_msg.toImageMsg());
        ROS_INFO("Prepare Segmentation Image");
        cv_bridge::CvImage segmentImage_msg;
        segmentImage_msg.encoding = sensor_msgs::image_encodings::TYPE_8UC3;
        segmentImage_msg.image    = SegmentMat;
        segmentImage_msg.header.seq = seq;
        segmentImage_msg.header.frame_id = hand_kp_pter->header.frame_id;;
        segmentImage_msg.header.stamp =  hand_kp_pter->header.stamp;
        segmentpublisher_.publish(segmentImage_msg.toImageMsg());

        ROS_INFO("Prepare Hand Detection Result");

        visualization_msgs::Marker bone;
        bone.header.frame_id = hand_kp_pter->header.frame_id;
        bone.header.stamp = hand_kp_pter->header.stamp;
        bone.ns = "finger_tracking";
        bone.type = visualization_msgs::Marker::LINE_LIST;
        bone.id = 0;
        bone.action = visualization_msgs::Marker::ADD;
        bone.pose.orientation.w = 1.0;
        bone.scale.x = 0.001;
        bone.color.a = 1.0;
        bone.color.g = 1.0;
        for(int finger = 0; finger <5; finger++){
            for(int i = 1; i< 5; i++){
                geometry_msgs::Point p2;
                p2.x = articulation.points[5*finger+i].x;
                p2.y = articulation.points[5*finger+i].y;
                p2.z = articulation.points[5*finger+i].z;
                bone.points.push_back(p2);
                p2.x = articulation.points[5*finger+i+1].x;
                p2.y = articulation.points[5*finger+i+1].y;
                p2.z = articulation.points[5*finger+i+1].z;
                bone.points.push_back(p2);
            }
        }
        for(int i = 0; i< 2; i++){
            for(int j = 0; j< 3; j++){
                geometry_msgs::Point p2;
                p2.x = articulation.points[6+5*j+i].x;
                p2.y = articulation.points[6+5*j+i].y;
                p2.z = articulation.points[6+5*j+i].z;
                bone.points.push_back(p2);
                p2.x = articulation.points[6+5*j+5+i].x;
                p2.y = articulation.points[6+5*j+5+i].y;
                p2.z = articulation.points[6+5*j+5+i].z;
                bone.points.push_back(p2);
            }
        }
        geometry_msgs::Point p2;
        p2.x = articulation.points[1].x;
        p2.y = articulation.points[1].y;
        p2.z = articulation.points[1].z;
        bone.points.push_back(p2);
        p2.x = articulation.points[6].x;
        p2.y = articulation.points[6].y;
        p2.z = articulation.points[6].z;
        bone.points.push_back(p2);
        bone_pub_.publish( bone );
        ///////////////////////
        visualization_msgs::Marker bone_leap;
        bone_leap.header.frame_id = hand_kp_pter->header.frame_id;
        bone_leap.header.stamp = hand_kp_pter->header.stamp;
        bone_leap.ns = "finger_tracking";
        bone_leap.type = visualization_msgs::Marker::LINE_LIST;
        bone_leap.id = 0;
        bone_leap.action = visualization_msgs::Marker::ADD;
        bone_leap.pose.orientation.w = 1.0;
        bone_leap.scale.x = 0.001;
        bone_leap.color.a = 1.0;
        bone_leap.color.r = 1.0;
        for(int finger = 0; finger <5; finger++){
            for(int i = 1; i< 5; i++){
                geometry_msgs::Point p;
                p.x = leap_articulation.points[5*finger+i].x;
                p.y = leap_articulation.points[5*finger+i].y;
                p.z = leap_articulation.points[5*finger+i].z;
                bone_leap.points.push_back(p);
                p.x = leap_articulation.points[5*finger+i+1].x;
                p.y = leap_articulation.points[5*finger+i+1].y;
                p.z = leap_articulation.points[5*finger+i+1].z;
                bone_leap.points.push_back(p);
            }
        }
        for(int i = 0; i< 2; i++){
            for(int j = 0; j< 3; j++){
                geometry_msgs::Point p;
                p.x = leap_articulation.points[6+5*j+i].x;
                p.y = leap_articulation.points[6+5*j+i].y;
                p.z = leap_articulation.points[6+5*j+i].z;
                bone_leap.points.push_back(p);
                p.x = leap_articulation.points[6+5*j+5+i].x;
                p.y = leap_articulation.points[6+5*j+5+i].y;
                p.z = leap_articulation.points[6+5*j+5+i].z;
                bone_leap.points.push_back(p);
            }
        }
        geometry_msgs::Point p;
        p.x = leap_articulation.points[1].x;
        p.y = leap_articulation.points[1].y;
        p.z = leap_articulation.points[1].z;
        bone_leap.points.push_back(p);
        p.x = leap_articulation.points[6].x;
        p.y = leap_articulation.points[6].y;
        p.z = leap_articulation.points[6].z;
        bone_leap.points.push_back(p);
        leap_articulate_pub_.publish( bone_leap );

        ros::Time time9 = ros::Time::now();
        std::cout<<"FPS: "<< time9-time0 << std::endl;
        ROS_INFO("One callback done");


    }
    catch (std::exception& e)
    {
        //if there is an error during conversion, display it
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

}


