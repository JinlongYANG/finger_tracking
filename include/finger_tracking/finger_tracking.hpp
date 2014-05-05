#ifndef slam_hpp
#define slam_hpp

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <algorithm>
#include <iostream>
#include <math.h>
#include <numeric>

//using std::vector;
//using cv::Mat;
//using cv::KeyPoint;

class Finger_tracking{
private:

public:

    virtual ~Finger_tracking(){}

    virtual void hand_seg(){
        return;
    }
};


//class Slam {

//private:
//    double min_depth_;
//    double max_depth_;
//    int line_requirement_;

//public:

//    Slam(double min_depth, double max_depth, int line_requirement){
//        min_depth_=min_depth;
//        max_depth_=max_depth;
//        line_requirement_=line_requirement;
//    }

//    virtual ~Slam(){}

//    virtual void match(const Mat& srcDescriptors, const Mat& destDescriptors,
//                       vector<DMatch>& matches ) const {
//        //        if((srcDescriptors.rows==0) || (destDescriptors.rows==0)){
//        //            DMatch failed;
//        //            failed.queryIdx = -1;             //srckeypointIndex
//        //            failed.trainIdx = -1;       //destKeypointIndex
//        //            failed.distance = -1;
//        //            matches.push_back(failed);
//        //            return;
//        //        }

//        cv::vector<float > difference;
//        Mat coherenceMat;
//        DMatch descrMatches;

//        coherenceMat.create(srcDescriptors.rows, destDescriptors.rows,CV_32FC1);

//        if (srcDescriptors.cols==destDescriptors.cols){

//            // find distances btw srcDescr and destDescr
//            for(int rowSrc= 0; rowSrc<srcDescriptors.rows; ++rowSrc){
//                for(int rowDest= 0; rowDest<destDescriptors.rows; ++rowDest){

//                    double distance;
//                    Mat sourceRow, destRow;
//                    //                    switch (line_requirement_) {
//                    //                        case 0://l1 norm
//                    //                            //distance = norm(srcDescriptors.row(rowSrc), destDescriptors.row(rowDest), NORM_L1);
//                    //                            absdiff(srcDescriptors.row(rowSrc), destDescriptors.row(rowDest), difference);
//                    //                            distance = std::accumulate(difference.begin(), difference.end(), 0);
//                    //                            break;
//                    //                        case 2://hamming distance
//                    //                            srcDescriptors.row(rowSrc).convertTo(sourceRow,CV_8U);
//                    //                            destDescriptors.row(rowDest).convertTo(destRow,CV_8U);
//                    //                            distance = norm(sourceRow,destRow, NORM_HAMMING);
//                    //                            //ROS_INFO("distance opencv hamming; %f", distance);
//                    //                            break;
//                    //                        default://l2 norm
//                    //distance = norm(srcDescriptors.row(rowSrc), destDescriptors.row(rowDest), NORM_L2);
//                    //ROS_INFO("distance with opencv norm= %f ",distance);
//                    absdiff(srcDescriptors.row(rowSrc), destDescriptors.row(rowDest), difference);
//                    distance = std::inner_product(difference.begin(), difference.end(), difference.begin(), 0);
//                    //ROS_INFO("distance with implemented norm= %f ",distance);
//                    //break;
//                    //                    }
//                    //rows and cols of coherence matrix shows srcKeypoint and destKeypoint indices respectively
//                    coherenceMat.at<float>(rowSrc,rowDest)=distance;
//                }
//            }
//        }


//        //compare each destKeypoint distance with srcKeypoint distance and find min
//        for(int row= 0; row<coherenceMat.rows; ++row){

//            float min_dist = coherenceMat.at<float>(row,0);
//            int col_index = 0;
//            for(int col = 0; col<coherenceMat.cols; ++col){

//                if(min_dist > coherenceMat.at<float>(row,col)){
//                    min_dist = coherenceMat.at<float>(row,col);
//                    col_index = col ;
//                }
//            }


//            descrMatches.queryIdx = row;             //srckeypointIndex
//            descrMatches.trainIdx = col_index;       //destKeypointIndex
//            descrMatches.distance = min_dist;
//            if(min_dist<200000.0)
//            matches.push_back(descrMatches);
//            else{
//                            DMatch failed;
//                            failed.queryIdx = -1;             //srckeypointIndex
//                            failed.trainIdx = -1;       //destKeypointIndex
//                            failed.distance = -1;
//                            matches.push_back(failed);
//                            return;
//            }


//        }

//    }


//    virtual void drawMatches(const Mat& srcImage, vector<KeyPoint>& srcKeypoints,
//                             const Mat& destImage, vector<KeyPoint>& destKeypoints,
//                             Mat& combinedImage) const {



//        combinedImage.create(srcImage.rows, srcImage.cols+destImage.cols ,srcImage.type());

//        srcImage.copyTo(combinedImage(cv::Rect(0,0,srcImage.cols,srcImage.rows)));
//        destImage.copyTo(combinedImage(cv::Rect(srcImage.cols-1,0,destImage.cols,destImage.rows)));

//        //ROS_INFO("drawing %d matches", matches.size());
//        if(destKeypoints.at(0).pt.x == -1 ||srcKeypoints.at(0).pt.x == -1)
//            return;
//        //draw line for matches
//        for (int i=0 ; i < srcKeypoints.size(); i++ ){
//            int radius = 15;

//            //draw matches if octaves are equal
//            //if(srcKeypoints.at(i).octave == destKeypoints.at(i).octave){
//            //draw keypoints
//            if(destKeypoints.at(i).pt.x != -1 && srcKeypoints.at(i).pt.x != -1){
//                circle(combinedImage,srcKeypoints.at(i).pt,(srcKeypoints.at(i).octave+1)*radius, CV_RGB(255, 255, 255), 1, 8, 0 );
//                circle(combinedImage,cv::Point(destKeypoints.at(i).pt.x+srcImage.cols ,
//                                               destKeypoints.at(i).pt.y ),(destKeypoints.at(i).octave+1)*radius,CV_RGB(255, 255, 255), 1, 8, 0 );
//                //draw matches
//                int linethickness = 1;
//                line(combinedImage,srcKeypoints.at(i).pt,
//                     cv::Point(destKeypoints.at(i).pt.x + srcImage.cols , destKeypoints.at(i).pt.y),
//                     CV_RGB(255, 255, 255), linethickness);
//                //draw directions
//                line(combinedImage,
//                     srcKeypoints.at(i).pt,
//                     cv::Point(srcKeypoints.at(i).pt.x+(srcKeypoints.at(i).octave+1)*radius*cos(srcKeypoints.at(i).angle),
//                               srcKeypoints.at(i).pt.y+(srcKeypoints.at(i).octave+1)*radius*sin(srcKeypoints.at(i).angle)),
//                     CV_RGB(255, 255, 255),2);
//                line(combinedImage,
//                     cv::Point(destKeypoints.at(i).pt.x+srcImage.cols ,
//                               destKeypoints.at(i).pt.y ),
//                     cv::Point(destKeypoints.at(i).pt.x+srcImage.cols+(destKeypoints.at(i).octave+1)*radius*cos(destKeypoints.at(i).angle),
//                               destKeypoints.at(i).pt.y+(destKeypoints.at(i).octave+1)*radius*sin(destKeypoints.at(i).angle)),
//                     CV_RGB(255, 255, 255),2);
//                //test of gradiant after rotation
//                //                line(combinedImage,
//                //                     srcKeypoints.at(i).pt,
//                //                     cv::Point(srcKeypoints.at(i).pt.x+(srcKeypoints.at(i).octave+1)*radius*cos(srcKeypoints.at(i).response),
//                //                               srcKeypoints.at(i).pt.y+(srcKeypoints.at(i).octave+1)*radius*sin(srcKeypoints.at(i).response)),
//                //                     CV_RGB(0, 255, 0),2);
//                //                line(combinedImage,
//                //                     cv::Point(destKeypoints.at(i).pt.x+srcImage.cols ,
//                //                                             destKeypoints.at(i).pt.y ),
//                //                     cv::Point(destKeypoints.at(i).pt.x+srcImage.cols+(destKeypoints.at(i).octave+1)*radius*cos(destKeypoints.at(i).response),
//                //                               destKeypoints.at(i).pt.y+(destKeypoints.at(i).octave+1)*radius*sin(destKeypoints.at(i).response)),
//                //                     CV_RGB(0, 255, 0),2);
//                //test end
//                //            }
//                //            else {
//                //                //draw keypoints
//                //                circle(combinedImage,srcKeypoints.at(i).pt,(srcKeypoints.at(i).octave+1)*radius, CV_RGB(255, 255, 0), 1, 8, 0 );
//                //                circle(combinedImage,cv::Point(destKeypoints.at(i).pt.x+srcImage.cols ,
//                //                                               destKeypoints.at(i).pt.y ),(destKeypoints.at(i).octave+1)*radius,CV_RGB(0, 255, 255), 1, 8, 0 );
//                //            }
//            }
//        }

//        //namedWindow("combined",CV_WINDOW_NORMAL );
//        //imshow("combined", combinedImage);
//        //waitKey(0);
//    }
//    virtual void drawtemporalMatches(const Mat& srcImage, vector<KeyPoint>& srcKeypoints,
//                             const Mat& destImage, vector<KeyPoint>& destKeypoints,
//                             Mat& combinedImage) const {


//        combinedImage.create(srcImage.rows, srcImage.cols+destImage.cols ,srcImage.type());

//        srcImage.copyTo(combinedImage(cv::Rect(0,0,srcImage.cols,srcImage.rows)));
//        destImage.copyTo(combinedImage(cv::Rect(srcImage.cols-1,0,destImage.cols,destImage.rows)));

//        //ROS_INFO("drawing %d matches", matches.size());
//        if(destKeypoints.at(0).pt.x == -1 ||srcKeypoints.at(0).pt.x == -1)
//            return;
//        //draw line for matches
//        for (int i=0 ; i < srcKeypoints.size(); i++ ){
//            int radius = 10;

//            //draw keypoints
//            if(destKeypoints.at(i).pt.x != -1 && srcKeypoints.at(i).pt.x != -1){
//                circle(combinedImage,srcKeypoints.at(i).pt,(srcKeypoints.at(i).octave+1)*radius, CV_RGB(255, 255, 255), 1, 8, 0 );
//                circle(combinedImage,destKeypoints.at(i).pt,(destKeypoints.at(i).octave+1)*radius,CV_RGB(0, 0, 0), 1, 8, 0 );

//                //draw matches
//                int linethickness = 4;
//                line(combinedImage,srcKeypoints.at(i).pt,
//                     destKeypoints.at(i).pt,
//                     CV_RGB(0, 0, 0), linethickness);


//            }
//        }


//    }
//};
#endif
