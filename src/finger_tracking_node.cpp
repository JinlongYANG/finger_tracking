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
//using namespace tf2;

float max_depth = 1.5;
float min_depth = 0.3;


Finger_tracking_Node::Finger_tracking_Node(ros::NodeHandle& nh):
    imageTransport_(nh),
    timeSynchronizer_(10)
  //reconfigureServer_(ros::NodeHandle(nh,"finger_tracking")),
  //transformListener_(buffer_, true)
  //reconfigureCallback_(boost::bind(&finger_tracking_Node::updateConfig, this, _1, _2))
{

    rgbCameraSubscriber_.subscribe(nh, "/camera/rgb/image_rect_color", 5);
    rgbCameraInfoSubscriber_.subscribe(nh, "/camera/rgb/camera_info", 5);
    depthCameraSubscriber_.subscribe(nh, "/camera/depth_registered/image_raw", 5);
    depthCameraInfoSubscriber_.subscribe(nh, "/camera/depth/camera_info", 5);
    pointCloud2_.subscribe(nh, "/camera/depth/points", 5);
    //pointCloud2_.subscribe(nh, "/camera/depth_registered/points", 5);
    leapMotion_.subscribe(nh, "/leap_data",1);


    timeSynchronizer_.connectInput(rgbCameraSubscriber_, depthCameraSubscriber_,rgbCameraInfoSubscriber_,depthCameraInfoSubscriber_, pointCloud2_, leapMotion_);

    bgrImagePublisher_ = imageTransport_.advertise("BGR_Image", 1);
    depthImagePublisher_ = imageTransport_.advertise("Depth_Image", 1);
    cloud_pub_ = nh.advertise<sensor_msgs::PointCloud2>("Hand_pcl",1);
    hkp_cloud_pub_ = nh.advertise<sensor_msgs::PointCloud2>("Hand_kp_cl",1);

    ROS_INFO("Here");
    timeSynchronizer_.registerCallback(boost::bind(&Finger_tracking_Node::syncedCallback, this, _1, _2, _3, _4, _5, _6));
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


void Finger_tracking_Node::syncedCallback(const ImageConstPtr& cvpointer_rgbImage,const ImageConstPtr& cvpointer_depthImage, const CameraInfoConstPtr& cvpointer_rgbInfo, const CameraInfoConstPtr& cvpointer_depthInfo, const PointCloud2ConstPtr& pclpointer_pointCloud2, const leap_msgs::Leap::ConstPtr& ptr_leap){


    cv_bridge::CvImagePtr cvpointer_rgbFrame, cvpointer_depthFrame;
    Mat BGRImage,DepthImage, DepthMat;

    pcl::PointCloud<pcl::PointXYZ> msg_pcl;
    pcl::PointCloud<pcl::PointXYZRGB> handcloud, hand1_kpt, hand2_kpt;

    HandKeyPoints hand_kpt;
    Hand_XYZRGB hand1_XYZRGB, hand2_XYZRGB;


    ROS_INFO("Callback begins");

    try
    {
        int seq = cvpointer_rgbInfo->header.seq;

        /*******************   get color image and depth image    *******************/
        cvpointer_rgbFrame = cv_bridge::toCvCopy(cvpointer_rgbImage);
        cvpointer_depthFrame = cv_bridge::toCvCopy(cvpointer_depthImage);

        ROS_INFO("current image seq: %d ",cvpointer_rgbInfo->header.seq);
        BGRImage=cvpointer_rgbFrame->image;
        cvtColor( BGRImage, BGRImage, CV_RGB2BGR);
        DepthMat=cvpointer_depthFrame->image;
        /*maximum depth: 3m*/
        DepthImage = DepthMat*0.33;

        /*******************   read in the leapmotion data   *******************/
        hand_kpt.set_Leap_Msg(ptr_leap);
        if(hand_kpt.hands_count != 0){
            std::cout<< hand_kpt.hands_count<<std::endl;
            hand1_XYZRGB.palm_center.x = hand_kpt.hand_position.at(0).x/1000.0;
            hand1_XYZRGB.palm_center.y = hand_kpt.hand_position.at(0).y/1000.0;
            hand1_XYZRGB.palm_center.z = hand_kpt.hand_position.at(0).z/1000.0;
            uint8_t r1 = 0, g1 = 255, b1 = 0;
            uint32_t rgb1 = ((uint32_t)r1 << 16 | (uint32_t)g1 << 8 | (uint32_t)b1);
            hand1_XYZRGB.palm_center.rgb = *reinterpret_cast<float*>(&rgb1);
            std::cout<<"hand1: palm center: "<<hand1_XYZRGB.palm_center.x<<" "<<hand1_XYZRGB.palm_center.y << " " << hand1_XYZRGB.palm_center.z<<std::endl;

            pcl::PointXYZRGB h1_fingertips;
            for(size_t i = 0; i < hand_kpt.fingertip_position.size ()/* && i < 5*/; ++i){
                h1_fingertips.x = hand_kpt.fingertip_position.at(i).x/1000.0;
                h1_fingertips.y = hand_kpt.fingertip_position.at(i).y/1000.0;
                h1_fingertips.z = hand_kpt.fingertip_position.at(i).z/1000.0;
                uint8_t rf = 0, gf = 150, bf = 150;
                uint32_t rgbf = ((uint32_t)rf << 16 | (uint32_t)gf << 8 | (uint32_t)bf);
                h1_fingertips.rgb = *reinterpret_cast<float*>(&rgbf);
                hand1_XYZRGB.fingertip_position.push_back(h1_fingertips);

            }

            if(hand_kpt.hands_count > 1){
                hand2_XYZRGB.palm_center.x = hand_kpt.hand_position.at(1).x/1000.0;
                hand2_XYZRGB.palm_center.y = hand_kpt.hand_position.at(1).y/1000.0;
                hand2_XYZRGB.palm_center.z = hand_kpt.hand_position.at(1).z/1000.0;
                uint8_t r2 = 0, g2 = 255, b2 = 0;
                uint32_t rgb2 = ((uint32_t)r2 << 16 | (uint32_t)g2 << 8 | (uint32_t)b2);
                hand2_XYZRGB.palm_center.rgb = *reinterpret_cast<float*>(&rgb2);
                std::cout<<"hand2: palm center: "<<hand2_XYZRGB.palm_center.x<<" "<<hand2_XYZRGB.palm_center.y << " " << hand2_XYZRGB.palm_center.z<<std::endl;
            }
        }

        /*******************   Leap motion to Xtion coordinate transform   *******************/
        if(hand_kpt.hands_count != 0){
            hand1_kpt.push_back(hand1_XYZRGB.palm_center);

            std::cout<<"Palm center: "<<hand1_kpt.at(0).x<<" "<<hand1_kpt.at(0).y << " " << hand1_kpt.at(0).z<<std::endl;
            for(size_t i = 0; i < hand1_XYZRGB.fingertip_position.size(); ++i){
                hand1_kpt.push_back(hand1_XYZRGB.fingertip_position.at(i));
                std::cout<<"finger "<<i<<": "<<hand1_kpt.at(i).x<<" "<<hand1_kpt.at(i).y << " " << hand1_kpt.at(i).z<<std::endl;
            }
            std::cout<<"Size: "<<hand1_kpt.size()<<std::endl;

            /*******************   get point cloud    *******************/
            fromROSMsg(*pclpointer_pointCloud2, msg_pcl);

            std::cout<<msg_pcl.points.size()<<std::endl;}

        /*******************   segment hand from the point cloud   *******************/
        pcl::PointXYZRGB p;
        uint8_t r = 255, g = r, b = r;
        uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);

        for (size_t i = 0; i < msg_pcl.points.size (); ++i){
            if(msg_pcl.points[i].z < max_depth && msg_pcl.points[i].z > min_depth){
                p.rgb = *reinterpret_cast<float*>(&rgb);
                p.x = msg_pcl.points[i].x;
                p.y = msg_pcl.points[i].y;
                p.z = msg_pcl.points[i].z;
                handcloud.push_back(p);
            }
        }



        //std::cout<<DepthImage<<std::endl;

        //        cv::imshow("RGB Image", BGRImage);
        //        cv::imshow("Depth Image", DepthImage);
        //        cv::waitKey();

        //___________________________________________________________________________
        //        //************* Chapter 2 Stereo Matching *********************//
        //        /////////////////////////////////////////////////////////////////
        //        vector<KeyPoint> matchedkeyPoints_inleft, matchedkeyPoints_inright;
        //        //ROS_INFO("min:%f, max:%f", min_disparity_, max_disparity_);

        //        vector<KeyPoint> keyPoints_inleft, keyPoints_inright;
        //        //1. get keypoints of both left and right image:


        //        //ROS_INFO("%ld left keypoints", keyPoints_inleft.size());
        //        //ROS_INFO("%ld right keypoints", keyPoints_inright.size());


        //        if(firstframe_flag_ == true){

        //            detector_->detect(leftGreyImage,keyPoints_inleft);
        //            detector_->detect(rightGreyImage,keyPoints_inright);
        //            firstframeprocessing(leftGreyImage, rightGreyImage, keyPoints_inleft, keyPoints_inright, extractor_, slam_, stereoCameraModel_,
        //                                 leftInfo, rightInfo,
        //                                 min_disparity_, max_disparity_, max_map_size_,
        //                                 map_,matchedkeyPoints_inleft,matchedkeyPoints_inright,
        //                                 cloud);
        //            //std::cout<<map_<<std::endl;

        //            slam_->drawMatches(leftGreyImage, matchedkeyPoints_inleft,
        //                               rightGreyImage, matchedkeyPoints_inright,
        //                               img_matches);


        //            //pcl::PointCloud<pcl::PointXYZ> cloud;
        //            //cloud.push_back(pcl::PointXYZ(keyPoint3D_c.x,keyPoint3D_c.y,keyPoint3D_c.z));

        //            cameraMatrix_.at<float>(0,0)=leftInfo->K[0];
        //            cameraMatrix_.at<float>(0,1)=leftInfo->K[1];
        //            cameraMatrix_.at<float>(0,2)=leftInfo->K[2];
        //            cameraMatrix_.at<float>(1,0)=leftInfo->K[3];
        //            cameraMatrix_.at<float>(1,1)=leftInfo->K[4];
        //            cameraMatrix_.at<float>(1,2)=leftInfo->K[5];
        //            cameraMatrix_.at<float>(2,2)=1;

        //            matchedkeyPoints_inleft_last_ = matchedkeyPoints_inleft;

        //            //accumulatedtransform_ *= estimateEigen_last_.inverse();

        //            //firstframe is a keyframe;
        //            Keyframeprocessing(keyframe_max_size_,keyframe_que_,leftGreyImage, rightGreyImage,
        //                               accumulatedtransform_que_, accumulatedtransform_,
        //                               last_Keyframe_keypoints_, matchedkeyPoints_inleft);

        //            matchedkeyPoints_inleft.clear();
        //            matchedkeyPoints_inright.clear();
        //            old_LeftKeyframeImage = leftGreyImage;


        //        }
        //        else if (seq %1 == 0){

        //            //stereo match -> outputs: last leftKP
        //            detector_->detect(leftGreyImage,keyPoints_inleft);
        //            detector_->detect(rightGreyImage,keyPoints_inright);

        //            //temporal match -> outputs: current leftKP
        //            vector<KeyPoint> temporalMatchedCurrentLeftKP, temporalMatchedLastleftKP;
        //            //            vector<KeyPoint> last_Keyframe_KPs;
        //            //            Mat last_Keyframe_Leftimage;

        //            temporalMatch(keyPoints_inleft, last_Keyframe_keypoints_,
        //                          temporalMatchedCurrentLeftKP, temporalMatchedLastleftKP,
        //                          old_LeftKeyframeImage, leftGreyImage, extractor_, slam_);

        //            slam_->drawtemporalMatches(leftGreyImage, temporalMatchedCurrentLeftKP,
        //                                       old_LeftKeyframeImage, temporalMatchedLastleftKP,
        //                                       img_matches);




        //            //std::cout<<"after stereo match: currLeftpts: "<<matchedkeyPoints_inleft.size()<<std::endl;
        //            //std::cout<<"after stereo match: lastLeftpts: "<<matchedkeyPoints_inleft_last_.size()<<std::endl;
        //            //std::cout<<"after temporal match: currLeftpts: "<<temporalMatchedCurrentLeftKP.size()<<std::endl;
        //            //std::cout<<"after temporal match: lastLeftpts: "<<temporalMatchedLastleftKP.size()<<std::endl;

        //            //3d-2d position calculations
        //            stereoCameraModel_.fromCameraInfo(leftInfo,rightInfo);
        //            cv::Point3d keyPoint3D;
        //            Mat lastKP_in3d, currKP_in2d;

        //            for(int i = 0; i < temporalMatchedCurrentLeftKP.size(); i++){
        //                float currKP_response = temporalMatchedCurrentLeftKP.at(i).response;
        //                float prevKP_response = temporalMatchedLastleftKP.at(i).response;

        //                if((currKP_response<min_disparity_) || (currKP_response>max_disparity_)
        //                        || (prevKP_response<min_disparity_) || (prevKP_response>max_disparity_)){

        //                    continue;
        //                }
        //                cv::Point2d leftUV=cv::Point2d(temporalMatchedLastleftKP.at(i).pt.x,temporalMatchedLastleftKP.at(i).pt.y);
        //                stereoCameraModel_.projectDisparityTo3d(leftUV, temporalMatchedLastleftKP.at(i).response, keyPoint3D);
        //                //ROS_INFO("pre %d: keyp(%f,%f) with disparity: %f estimated at (%lf,%lf,%lf)",validkeypoint_, leftUV.x,leftUV.y,temporalMatchedLastleftKP.at(i).response,keyPoint3D_p.x,keyPoint3D_p.y,keyPoint3D_p.z);

        //                //cloud.push_back(pcl::PointXYZ(keyPoint3D.x,keyPoint3D.y,keyPoint3D.z));
        //                //stereoMatchedLeftKeyPoints.push_back( temporalMatchedLastleftKP.at(i));
        //                //stereoMatchedRightKeyPoints.push_back(unfilteredstereoMatchedRightKeyPoints.at(i));

        //                Mat point;
        //                point.create(1,3,CV_32F);
        //                point.at<float>(0,0)=keyPoint3D.x;
        //                point.at<float>(0,1)=keyPoint3D.y;
        //                point.at<float>(0,2)=keyPoint3D.z;

        //                lastKP_in3d.push_back(point);

        //                Mat newpoint;
        //                newpoint.create(1,2,CV_32F);
        //                newpoint.at<float>(0,0)=temporalMatchedCurrentLeftKP.at(i).pt.x;
        //                newpoint.at<float>(0,1)=temporalMatchedCurrentLeftKP.at(i).pt.y;

        //                currKP_in2d.push_back(newpoint);
        //            }

        //            //pose estimation by PnP Ransac
        //            Mat Rotation_vector, Translation_vector, R, t;
        //            vector<float> distortionCoeffs;

        //            bool useExtrinsicGuess=false;
        //            int iterationsCount=200;
        //            float reprojectionError = 8;
        //            int minInliersCount=100;
        //            Mat inliers= Mat::zeros(1,100, CV_32FC1);

        //            cv::solvePnPRansac(lastKP_in3d, currKP_in2d, cameraMatrix_, distortionCoeffs,
        //                               Rotation_vector, Translation_vector, useExtrinsicGuess,
        //                               iterationsCount,reprojectionError,minInliersCount,inliers, CV_EPNP );

        //            cv::Rodrigues(Rotation_vector, R);
        //            t = Translation_vector;
        //            //inliers ratio
        //            inliers_ratio = (float)inliers.rows/lastKP_in3d.rows;

        //            //std::cout<<"Rotation "<< R<<std::endl;
        //            //std::cout<<"translation"<<t<<std::endl;
        //            std::cout<<"keypoints: "<<lastKP_in3d.rows<<" inliers: "<<inliers.rows<<" inliers ratio: "<<inliers_ratio<<std::endl;

        //            //Broadcasting Transformation
        //            //use tf2 to publish current pose estimate
        //            Eigen::Matrix4f estimateEigen;
        //            Mat estimateCv;
        //            bool RT_flag;
        //            float threshold = 0.2;
        //            float t_limit = 12.0;

        //            if(threshold > inliers_ratio || t.dot(t) > t_limit){
        //                RT_flag = false;
        //                std::cout<<"translation "<<t.dot(t)<<std::endl;
        //                ROS_ERROR("inlier ratio is too low or translation is too large");
        //            }
        //            else {
        //                RT_flag =true;
        //            }

        //            if (RT_flag==true) {
        //                estimateCv=Mat::zeros(4,4,CV_32F);
        //                R.copyTo(estimateCv(cv::Rect(0,0,3,3)));
        //                t.copyTo(estimateCv(cv::Rect(3,0,1,3)));
        //                estimateCv.at<float>(3,3)=1;
        //                cv2eigen(estimateCv,estimateEigen);
        //                prev_t=t;

        //                estimateEigen_last_ =estimateEigen;
        //            }
        //            else{
        //                t =prev_t;
        //                estimateEigen = estimateEigen_last_;
        //            }



        //            //keyframe selection criteria
        //            float ratio_th = 0.15;
        //            float distance_th = 1.5;
        //            float angle_th = 1.0;
        //            //calcullate parameters for selection
        //            distancetravelled = sqrt(t.dot(t));
        //            Mat projMatrix = estimateCv, cameraMatrix, rotMatrix, transVect, eulerAngles;
        //            projMatrix.pop_back();

        //            decomposeProjectionMatrix(projMatrix, cameraMatrix, rotMatrix, transVect,
        //                                      noArray(), noArray(), noArray(), eulerAngles);
        //            anglerotated = eulerAngles.at<double>(0,1);

        //            ROS_INFO("*******************anglerotated: %f", anglerotated);

        //            ROS_INFO("*************distance travelled: %f",distancetravelled);

        //            if (distancetravelled > distance_th || abs(anglerotated) > angle_th){
        //                //keyframe;
        //                firstframeprocessing(leftGreyImage, rightGreyImage, keyPoints_inleft, keyPoints_inright, extractor_, slam_, stereoCameraModel_,
        //                                     leftInfo, rightInfo,
        //                                     min_disparity_, max_disparity_, max_map_size_,
        //                                     map_,matchedkeyPoints_inleft,matchedkeyPoints_inright,
        //                                     cloud);
        //                matchedkeyPoints_inleft_last_ = matchedkeyPoints_inleft;

        //                Keyframeprocessing(keyframe_max_size_,keyframe_que_,leftGreyImage, rightGreyImage,
        //                                   accumulatedtransform_que_, accumulatedtransform_,
        //                                   last_Keyframe_keypoints_, matchedkeyPoints_inleft);
        //                accumulatedtransform_= accumulatedtransform_*estimateEigen.inverse();


        //                //dense stereo matching
        //                sbm.operator()(leftGreyImage, rightGreyImage, disparity_Map, CV_32F );
        //                //std::cout<<disparity_Map<<std::endl;

        //                ROS_INFO("rows: %d; cols: %d", leftGreyImage.rows, leftGreyImage.cols);
        //                ROS_INFO("rows: %d; cols: %d", disparity_Map.rows, disparity_Map.cols);

        //                for(int i = 0; i < disparity_Map.rows; i++){
        //                    for(int j = 0; j < disparity_Map.cols; j++){
        //                        if(disparity_Map.at<float>(i,j)>0){
        //                            cv::Point2d leftUV=cv::Point2d(j,i);
        //                            cv::Point3d p3d;
        //                            pcl::PointXYZRGB p;
        //                            stereoCameraModel_.projectDisparityTo3d(leftUV, disparity_Map.at<float>(i,j), p3d);
        //                            if(p3d.z<75){
        //                                uint8_t r = leftGreyImage.at<unsigned char>(i,j), g = r, b = r;
        //                                uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
        //                                p.rgb = *reinterpret_cast<float*>(&rgb);
        //                                p.x = p3d.x;
        //                                p.y = p3d.y;
        //                                p.z = p3d.z;
        //                                cloud.push_back(p);
        //                            }

        //                        }
        //                    }

        //                }



        //                old_LeftKeyframeImage = leftGreyImage;

        //                ROS_INFO("!!!!!!!!!added keyframe!!!!!!!!!");
        //            }


        //        }
        //        ROS_INFO("keyframe que size : %ld", keyframe_que_.size());
        //        ROS_INFO("accumulated que size : %ld", accumulatedtransform_que_.size());


        //        //std::cout<<"accMatrix after:  "<<std::endl<<accumulatedtransform_<<std::endl;
        //        //message preparation
        //        Eigen::Quaternionf quaternion(accumulatedtransform_.block<3,3>(0,0));
        //        geometry_msgs::Transform transform;

        //        transform.translation.x=accumulatedtransform_(0,3);
        //        transform.translation.y=accumulatedtransform_(1,3);
        //        transform.translation.z=accumulatedtransform_(2,3);

        //        transform.rotation.x=quaternion.x();
        //        transform.rotation.y=quaternion.y();
        //        transform.rotation.z=quaternion.z();
        //        transform.rotation.w=quaternion.w();

        //        //ROS_INFO("transform x:%f y:%f z:%f",accumulatedtransform_(0,3),accumulatedtransform_(1,3),accumulatedtransform_(2,3));

        //        transformStamped_.header.stamp=leftInfo->header.stamp;
        //        transformStamped_.header.frame_id="start";
        //        transformStamped_.child_frame_id="current";
        //        transformStamped_.transform=transform;

        //        transformBroadcaster_.sendTransform(transformStamped_);

        //        //publish the estimated and groundTruth path
        //        geometry_msgs::PoseStamped currentPoseStamped;
        //        currentPoseStamped.header.stamp=leftInfo->header.stamp;
        //        currentPoseStamped.pose.position.x=transform.translation.x;
        //        currentPoseStamped.pose.position.y=transform.translation.y;
        //        currentPoseStamped.pose.position.z=transform.translation.z;

        //        currentPoseStamped.pose.orientation=transform.rotation;

        //        estimatedPath_msg_.header.frame_id = "/start";
        //        estimatedPath_msg_.poses.push_back(currentPoseStamped);
        //        estimatedPath_pub_.publish(estimatedPath_msg_);


        //        geometry_msgs::TransformStamped transformStamped = buffer_.Buffer::lookupTransform("start", "kitty_stereo/left", ros::Time(0), ros::Duration(10));


        //        currentPoseStamped.header.stamp=leftInfo->header.stamp;
        //        currentPoseStamped.pose.position.x=transformStamped.transform.translation.x;
        //        currentPoseStamped.pose.position.y=transformStamped.transform.translation.y;
        //        currentPoseStamped.pose.position.z=transformStamped.transform.translation.z;
        //        currentPoseStamped.pose.orientation=transformStamped.transform.rotation;

        //        groundTruthPath_msg_.header.frame_id = "/start";
        //        groundTruthPath_msg_.poses.push_back(currentPoseStamped);
        //        groundTruthPath_pub_.publish(groundTruthPath_msg_);




        //        //publish pointCloud
        //        sensor_msgs::PointCloud2 cloud_msg;
        //        toROSMsg(cloud,cloud_msg);
        //        cloud_msg.header.frame_id=stereoCameraModel_.tfFrame();//leftInfo->header.frame_id;
        //        cloud_pub_.publish(cloud_msg);

        //________________________________________________________________________________________________________________


        /*******************   Convert the CvImage to a ROS image message and publish it to topics.   *******************/
        cv_bridge::CvImage bgrImage_msg;
        bgrImage_msg.encoding = sensor_msgs::image_encodings::BGR8;
        bgrImage_msg.image    = BGRImage;
        bgrImage_msg.header.seq = seq;
        bgrImage_msg.header.frame_id = seq;
        bgrImage_msg.header.stamp = ros::Time::now();
        bgrImagePublisher_.publish(bgrImage_msg.toImageMsg());

        cv_bridge::CvImage depthImage_msg;
        depthImage_msg.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
        depthImage_msg.image    = DepthImage;
        depthImage_msg.header.seq = seq;
        depthImage_msg.header.frame_id = seq;
        depthImage_msg.header.stamp = ros::Time::now();
        depthImagePublisher_.publish(depthImage_msg.toImageMsg());

        /*******************   publish pointCloud   *******************/
        sensor_msgs::PointCloud2 cloud_msg;
        toROSMsg(handcloud,cloud_msg);
        cloud_msg.header.frame_id=cvpointer_depthInfo->header.frame_id;
        cloud_pub_.publish(cloud_msg);

        sensor_msgs::PointCloud2 hand1_kpt_msg;
        toROSMsg(hand1_kpt,hand1_kpt_msg);
        hand1_kpt_msg.header.frame_id=cvpointer_depthInfo->header.frame_id;
        hkp_cloud_pub_.publish(hand1_kpt_msg);

        /*******************   clear data   *******************/
        hand_kpt.Clear();


        ROS_INFO("One callback done");


    }
    catch (std::exception& e)
    {
        //if there is an error during conversion, display it
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

}
