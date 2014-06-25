#ifndef POSEESTIMATE_HPP_INCLUDED
#define POSEESTIMATE_HPP_INCLUDED
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
//#include <opencv2/core/eigen.hpp>
//#include <Geometry>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>


using namespace cv;
using namespace std;
namespace poseEstimate{

void mean(Mat keypoints, Mat& miu_){
    int c = keypoints.cols;
    miu_=Mat::zeros(keypoints.rows,1,keypoints.type());
    for (int i=0; i<c ;++i){
        miu_ +=keypoints.col(i);
    }

    miu_ = miu_/c;

}

void variance(Mat keypoints,Mat miu_, float& var_){
    int c = keypoints.cols;
    var_ = 0.0;
    for (int i=0; i<c ;++i){
        Mat m = keypoints.col(i) - miu_;
        var_ += m.dot(m);
    }
    var_ = var_/c;
}

void covariance(Mat keypoints1, Mat keypoints2, Mat miu_cr, Mat miu_pr, Mat& cov_){
    int c = keypoints1.cols;
    cov_= Mat::zeros(3,3,keypoints1.type());
    for (int i=0; i<c ;++i){
        Mat m=(keypoints1.col(i) - miu_cr);
        Mat n=(keypoints2.col(i) - miu_pr);
        cov_ += m*n.t();
    }
    cov_ = cov_/c;

}


class poseestimate
{
public:

    static void compute(Mat A, Mat B, Mat& Rotation, Mat& Translation){
        //inputs: 3DKeypoints at time step t-1 and t in shape [3xn]
        //outputs: Rotation,translation
        //ROS_INFO("poseestimation");
        //check if keypointMat size match!
        if(A.size() == B.size()){
            Mat miu_cr, miu_pr, cov_;
            float var_cr, var_pr;

            //compute means
            mean(A,miu_cr);
            mean(B,miu_pr);
            //compute variances
            variance(B,miu_pr,var_pr);
            //compute covariance and SVD(covariance)
            covariance(A,B,miu_cr,miu_pr,cov_);
            Mat U,D,VTr;
            Mat S;
            SVD::compute(cov_,D,U,VTr,0);

            int m=cov_.rows;
            //determine D
            Mat d =d.zeros(m,m,A.type());
            for(int i= 0; i<m; ++i){
                d.at<float>(i,i)=D.at<float>(i,0);
            }


            if ( determinant(cov_) >= 0 ){
                S = Mat::eye(m,m,A.type());
            } else {
                S = Mat::eye(m,m,A.type());
                S.at<float>(m-1,m-1)=-1;
            }


            //compute rotation,translation and scale
            float Scale;

            Scalar t = trace(d*S);
            Scale = t[0]/var_pr;
            Rotation = U * S * VTr;
            Translation = miu_cr - Scale*Rotation*miu_pr;
            //cout<<"Rotation: \n"<<Rotation<<endl;
            //cout<<"Translation: "<<Translation<<endl;

        }

    }
private:
protected:



};
}

#endif // POSEESTIMATE_HPP_INCLUDED
