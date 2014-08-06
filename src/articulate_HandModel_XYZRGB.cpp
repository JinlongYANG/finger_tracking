#include "finger_tracking/articulate_HandModel_XYZRGB.h"
#include <opencv2/calib3d/calib3d.hpp>
#include "finger_tracking/Ransac.h"
#include "finger_tracking/poseestimate.hpp"
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

#define PI 3.14159265

using namespace cv;


float parameters_max[26] = {10, 10, 10,
                            180, 180, 180,
                            80, 10, 70, 90,
                            30, 30, 110, 80,
                            30, 30, 110, 80,
                            30, 30, 110, 80,
                            30, 30, 110, 80,};

float parameters_min[26] = {0, 0, 0,
                            -180, -180, -180,
                            10, -80, 0, -10,
                            -30, -90, 0, 0,
                            -30, -90, 0, 0,
                            -30, -90, 0, 0,
                            -30, -90, 0, 0};

////////////////////////////////////////////////////////////
//*************      usefull functions     ***************//
Mat R_x(float theta){
    Mat Rx = Mat::zeros(3,3,CV_32FC1);
    Rx.at<float>(0,0) = 1;
    Rx.at<float>(1,1) = cos(theta*PI/180.0);
    Rx.at<float>(1,2) = -sin(theta*PI/180.0);
    Rx.at<float>(2,1) = sin(theta*PI/180.0);
    Rx.at<float>(2,2) = cos(theta*PI/180.0);
    return Rx;
}

Mat R_y(float theta){
    Mat Ry = Mat::zeros(3,3,CV_32FC1);
    Ry.at<float>(0,0) = cos(theta*PI/180.0);
    Ry.at<float>(0,2) = sin(theta*PI/180.0);
    Ry.at<float>(1,1) = 1;
    Ry.at<float>(2,0) = -sin(theta*PI/180.0);
    Ry.at<float>(2,2) = cos(theta*PI/180.0);
    return Ry;
}

Mat R_z(float theta){
    Mat Rz = Mat::zeros(3,3,CV_32FC1);
    Rz.at<float>(0,0) = cos(theta*PI/180.0);
    Rz.at<float>(0,1) = -sin(theta*PI/180.0);
    Rz.at<float>(1,0) = sin(theta*PI/180.0);
    Rz.at<float>(1,1) = cos(theta*PI/180.0);
    Rz.at<float>(2,2) = 1;
    return Rz;
}

float Distance_2XYZRGB(pcl::PointXYZRGB p1, pcl::PointXYZRGB p2){
    float dis = sqrt((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y)+(p1.z-p2.z)*(p1.z-p2.z));
    if (dis == 0)
        return 0.00000001;
    else
        return dis;
}

float Distance_XYZMat(pcl::PointXYZ p1, Mat p2){
    float dis = sqrt((p1.x-p2.at<float>(0,0))*(p1.x-p2.at<float>(0,0))+(p1.y-p2.at<float>(1,0))*(p1.y-p2.at<float>(1,0))+(p1.z-p2.at<float>(2,0))*(p1.z-p2.at<float>(2,0)));
    if (dis == 0)
        return 0.00000001;
    else
        return dis;
}

float Distance_XYZXYZRGB(pcl::PointXYZ p1, pcl::PointXYZRGB p2){
    float dis = sqrt((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y)+(p1.z-p2.z)*(p1.z-p2.z));
    if (dis == 0)
        return 0.00000001;
    else
        return dis;
}

float Distance_2XYZ(pcl::PointXYZ p1, pcl::PointXYZ p2){
    float dis = sqrt((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y)+(p1.z-p2.z)*(p1.z-p2.z));
    if (dis == 0)
        return 0.00000001;
    else
        return dis;
}

float Distance_2Point3d(Point3d p1, Point3d p2){
    float dis = sqrt((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y)+(p1.z-p2.z)*(p1.z-p2.z));
    if (dis == 0)
        return 0.00000001;
    else
        return dis;
}

float Distance_XYZPoint3d(pcl::PointXYZ p1, Point3d p2){
    float dis = sqrt((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y)+(p1.z-p2.z)*(p1.z-p2.z));
    if (dis == 0)
        return 0.00000001;
    else
        return dis;
}

float arc2degree(float arc){
    return 180.0*arc/PI;
}

float degree2arc(float degree){
    return degree/180.0*PI;
}

/////////////////////////////////////////////////////////////////

articulate_HandModel_XYZRGB::articulate_HandModel_XYZRGB()
{
    //1. joints color initialization:
    joints_position[0].rgb =  ((uint32_t)255 << 16 | (uint32_t)255 << 8 | (uint32_t)255);

    for(int finger_index = 0; finger_index < 5; finger_index ++){
        uint8_t rf = 63*finger_index;
        uint8_t bf = 255-rf;
        for(int j = 0; j<5;j++){
            uint8_t gf = j*50;
            uint32_t rgbf = ((uint32_t)rf << 16 | (uint32_t)gf << 8 | (uint32_t)bf);
            joints_position[finger_index*5+j+1].rgb = *reinterpret_cast<float*>(&rgbf);
            joints_position[finger_index*5+j+1].x = (finger_index-2)/15.0*(j*0.3+1);
            joints_position[finger_index*5+j+1].y = (j-2)/15.0;
            joints_position[finger_index*5+j+1].z = 0.0;
        }
    }
    joints_position[5] = joints_position[4];

    //2. bone length initialization:
    //thumb
    bone_length[0][0] = 50.1779;
    bone_length[0][1] = 31.9564;
    bone_length[0][2] = 22.9945;
    bone_length[0][3] =0;
    //index finger
    bone_length[1][0] = 78.4271;
    bone_length[1][1] = 46.0471;
    bone_length[1][2] = 26.7806;
    bone_length[1][3] = 19.517;
    //middle finger
    bone_length[2][0] = 74.5294;
    bone_length[2][1] = 50.4173;
    bone_length[2][2] = 32.1543;
    bone_length[2][3] = 22.2665;
    //ring finger
    bone_length[3][0] = 67.2215;
    bone_length[3][1] = 46.8076;
    bone_length[3][2] = 31.4014;
    bone_length[3][3] = 22.1557;
    //pinky finger
    bone_length[4][0] = 62.4492;
    bone_length[4][1] = 32.2519;
    bone_length[4][2] = 21.0526;
    bone_length[4][3] = 18.672;

    //3. Model joints position initialization
    for(int i = 0; i < 26; i++){
        Model_joints[i] = Mat::zeros(3,1,CV_32FC1);
    }
    //3.1. palm joints: 1,6,11,16,21,7,12,17,22
    //palm joints with reference to palm/hand coordinate:
    //palm.thumb
    Model_joints[1].at<float>(0,0) = -0.019;
    Model_joints[1].at<float>(1,0) = -0.055;
    Model_joints[1].at<float>(2,0) = 0.001;
    //palm.index
    Model_joints[6].at<float>(0,0) = -0.014;
    Model_joints[6].at<float>(1,0) = -0.053;
    Model_joints[6].at<float>(2,0) = -0.008;

    Model_joints[7].at<float>(0,0) = -0.024;
    Model_joints[7].at<float>(1,0) = 0.019;
    Model_joints[7].at<float>(2,0) = 0;
    //palm.middle
    Model_joints[11].at<float>(0,0) = 0;
    Model_joints[11].at<float>(1,0) = -0.05;
    Model_joints[11].at<float>(2,0) = -0.008;

    Model_joints[12].at<float>(0,0) = -0.002;
    Model_joints[12].at<float>(1,0) = 0.023;
    Model_joints[12].at<float>(2,0) = 0;
    //palm.ring
    Model_joints[16].at<float>(0,0) = 0.014;
    Model_joints[16].at<float>(1,0) = -0.051;
    Model_joints[16].at<float>(2,0) = -0.008;

    Model_joints[17].at<float>(0,0) = 0.020;
    Model_joints[17].at<float>(1,0) = 0.018;
    Model_joints[17].at<float>(2,0) = 0;
    //palm.pinky
    Model_joints[21].at<float>(0,0) = 0.027;
    Model_joints[21].at<float>(1,0) = -0.053;
    Model_joints[21].at<float>(2,0) = -0.004;

    Model_joints[22].at<float>(0,0) = 0.042;
    Model_joints[22].at<float>(1,0) = 0.013;
    Model_joints[22].at<float>(2,0) = 0;

    for(int i = 0; i < 5; i++){
        virtual_joints[i] = Mat::zeros(3,1,CV_32FC1);
    }

    virtual_joints[0] = R_y(70)*Model_joints[1];

    virtual_joints[1].at<float>(0,0) = Model_joints[7].at<float>(0,0);
    virtual_joints[1].at<float>(1,0) = Model_joints[7].at<float>(1,0);
    virtual_joints[1].at<float>(2,0) = Model_joints[7].at<float>(2,0) + 0.1;

    virtual_joints[2].at<float>(0,0) = Model_joints[12].at<float>(0,0);
    virtual_joints[2].at<float>(1,0) = Model_joints[12].at<float>(1,0);
    virtual_joints[2].at<float>(2,0) = Model_joints[12].at<float>(2,0) + 0.1;

    virtual_joints[3].at<float>(0,0) = Model_joints[17].at<float>(0,0);
    virtual_joints[3].at<float>(1,0) = Model_joints[17].at<float>(1,0);
    virtual_joints[3].at<float>(2,0) = Model_joints[17].at<float>(2,0) + 0.1;

    virtual_joints[4].at<float>(0,0) = Model_joints[22].at<float>(0,0);
    virtual_joints[4].at<float>(1,0) = Model_joints[22].at<float>(1,0);
    virtual_joints[4].at<float>(2,0) = Model_joints[22].at<float>(2,0) + 0.1;

    //3.2.fingers:
    //3.2.1 index(extrinsic):
    Model_joints[8].at<float>(1,0) = bone_length[1][1]/1000.0;
    Model_joints[9].at<float>(1,0) = bone_length[1][2]/1000.0;
    Model_joints[10].at<float>(1,0) = bone_length[1][3]/1000.0;

    //    Model_joints[10] = Model_joints[10]+Model_joints[9]+Model_joints[8]+Model_joints[7];
    //    Model_joints[9] = Model_joints[9]+Model_joints[8]+Model_joints[7];
    //    Model_joints[8] = Model_joints[8]+Model_joints[7];

    //3.2.2 middel to pinky(extrinsic):
    for ( int i = 0; i < 3; ++i){
        Model_joints[i*5+13].at<float>(1,0) = bone_length[2+i][1]/1000;
        Model_joints[i*5+14].at<float>(1,0) = bone_length[2+i][2]/1000;
        Model_joints[i*5+15].at<float>(1,0) = bone_length[2+i][3]/1000;

        //        Model_joints[i*5+15] = Model_joints[i*5+15]+Model_joints[i*5+14]+Model_joints[i*5+13]+Model_joints[i*5+12];
        //        Model_joints[i*5+14] = Model_joints[i*5+14]+Model_joints[i*5+13]+Model_joints[i*5+12];
        //        Model_joints[i*5+13] = Model_joints[i*5+13]+Model_joints[i*5+12];

    }

    //3.2.3 thumb(extrinsic)
    Model_joints[2].at<float>(1,0) = bone_length[0][0]/1000.0;
    Model_joints[3].at<float>(1,0) = bone_length[0][1]/1000.0;
    Model_joints[4].at<float>(1,0) = bone_length[0][2]/1000.0;

    //    Model_joints[4] = Model_joints[4]+Model_joints[3]+Model_joints[2]+Model_joints[1];
    //    Model_joints[3] = Model_joints[3]+Model_joints[2]+Model_joints[1];
    //    Model_joints[2] = Model_joints[2]+Model_joints[1];
    //    Model_joints[4].copyTo(Model_joints[5]);

    palm_model = Mat::zeros(3, 9, CV_32FC1);

    Model_joints[6].copyTo(palm_model.col(0));
    Model_joints[7].copyTo(palm_model.col(1));
    Model_joints[11].copyTo(palm_model.col(2));
    Model_joints[12].copyTo(palm_model.col(3));
    Model_joints[16].copyTo(palm_model.col(4));
    Model_joints[17].copyTo(palm_model.col(5));
    Model_joints[21].copyTo(palm_model.col(6));
    Model_joints[22].copyTo(palm_model.col(7));
    Model_joints[1].copyTo(palm_model.col(8));

}

bool articulate_HandModel_XYZRGB::check_parameters(int &wrong_parameter_index){
    for(int i = 0; i < 26; i++){
        if(parameters[i] > parameters_max[i] || parameters[i] < parameters_min[i]){
            wrong_parameter_index = i;
            std::cout << "Wrong parameter index: " << wrong_parameter_index <<"; Value: "<< parameters[i] << std::endl;
            return false;
        }
    }
    return true;
}

void articulate_HandModel_XYZRGB::set_parameters(){
    //0: hand center position x;
    //1: hand center position y;
    //2: hand center position z;
    parameters[0] = 0;
    parameters[1] = 0;
    parameters[2] = 0;
    //3: palm rotation angle around x axis;
    //4: palm rotation angle around y axis;
    //5: palm rotation angle around z axis;
    parameters[3] = -20;
    parameters[4] = 0;
    parameters[5] = 0;
    //6: horizontal angle between thumb metacarpal and palm;
    //7: vertical angle between thumb metacarpal and palm;
    //8: angle between thumb metacarpal and proximal;
    //9: angle between thumb proximal and distal;
    parameters[6] = 20;
    parameters[7] = 0;
    parameters[8] = 0;
    parameters[9] = 0;
    //10: horizontal angle between index finger proximal and palm;
    //11: vertical angle between index finger proximal and palm;
    //12: angle between index finger proximal and intermediate;
    //13: angle between index finger intermediate and distal;
    parameters[10] = 10;
    parameters[11] = 0;
    parameters[12] = 0;
    parameters[13] = 0;
    //14: horizontal angle between middle finger proximal and palm;
    //15: vertical angle between middle finger proximal and palm;
    //16: angle between middle finger proximal and intermediate;
    //17: angle between middle finger intermediate and distal;
    parameters[14] = 0;
    parameters[15] = 0;
    parameters[16] = 0;
    parameters[17] = 1;
    //18: horizontal angle between ring finger proximal and palm;
    //19: vertical angle between ring finger proximal and palm;
    //20: angle between ring finger proximal and intermediate;
    //21: angle between ring finger intermediate and distal;
    parameters[18] = -10;
    parameters[19] = 0.234;
    parameters[20] = 0;
    parameters[21] = 0;
    //22: horizontal angle between pinky proximal and palm;
    //23: vertical angle between pinky proximal and palm;
    //24: angle between pinky proximal and intermediate;
    //25: angle between pinky intermediate and distal;
    parameters[22] = -25;
    parameters[23] = 0;
    parameters[24] = 1;
    parameters[25] = 1;
}

void articulate_HandModel_XYZRGB::get_parameters(){

    //1. find hand roll pitch yaw(parameter 3, 4, 5)
    //1.1 determin the translation matrix of palm;
    Mat palm;
    palm = Mat::zeros(3, 9, CV_32FC1);

    palm.at<float>(0,0) = joints_position[1].x;
    palm.at<float>(1,0) = joints_position[1].y;
    palm.at<float>(2,0) = joints_position[1].z;

    palm.at<float>(0,1) = joints_position[6].x;
    palm.at<float>(1,1) = joints_position[6].y;
    palm.at<float>(2,1) = joints_position[6].z;

    palm.at<float>(0,2) = joints_position[7].x;
    palm.at<float>(1,2) = joints_position[7].y;
    palm.at<float>(2,2) = joints_position[7].z;

    palm.at<float>(0,3) = joints_position[11].x;
    palm.at<float>(1,3) = joints_position[11].y;
    palm.at<float>(2,3) = joints_position[11].z;

    palm.at<float>(0,4) = joints_position[12].x;
    palm.at<float>(1,4) = joints_position[12].y;
    palm.at<float>(2,4) = joints_position[12].z;

    palm.at<float>(0,5) = joints_position[16].x;
    palm.at<float>(1,5) = joints_position[16].y;
    palm.at<float>(2,5) = joints_position[16].z;

    palm.at<float>(0,6) = joints_position[17].x;
    palm.at<float>(1,6) = joints_position[17].y;
    palm.at<float>(2,6) = joints_position[17].z;

    palm.at<float>(0,7) = joints_position[21].x;
    palm.at<float>(1,7) = joints_position[21].y;
    palm.at<float>(2,7) = joints_position[21].z;

    palm.at<float>(0,8) = joints_position[22].x;
    palm.at<float>(1,8) = joints_position[22].y;
    palm.at<float>(2,8) = joints_position[22].z;

    Mat joints_position_Mat = Mat::zeros(3,26, CV_32FC1);
    for ( int i = 0; i < 26; i++ ) {
        joints_position_Mat.at<float>(0,i) = joints_position[i].x;
        joints_position_Mat.at<float>(1,i) = joints_position[i].y;
        joints_position_Mat.at<float>(2,i) = joints_position[i].z;
    }

    Mat R,t;
    poseEstimate::poseestimate::compute(palm,palm_model,R,t);

    //1.2 get the angles:
    cv::Mat mtxR, mtxQ;
    cv::Vec3d angles;
    angles = cv::RQDecomp3x3(R, mtxR, mtxQ);

    parameters[3] = angles[0];
    parameters[4] = angles[1];
    parameters[5] = angles[2];

    //1.3 derotate the hand:
    for(int i = 0; i<26; i++){
        joints_position_Mat.col(i) = R.inv()*joints_position_Mat.col(i);
    }

    //2. find angles of every joints of index finger:
    ////////////////////////////////////////////
    //   /cos(r) -sin(r)  0  \ /1    0    0  \    / cr    -casr    sasr\   / 0 \    / x2 \
    //   |                   | |             |    |                    |  |     |   |    |
    //   |sin(r) cos(r)   0  |*|0    ca   -sa| =  |sr      cacr   -sacr|  |  y1 | = | y2 |
    //   |                   | |             |    |                    |  |     |   |    |
    //   \  0      0     1   / \0    sa   ca /    \  0      sa      ca /   \ 0 /    \ z2 /
    //
    //2.1 parameter 10, 11:
    Mat temp = joints_position_Mat.col(8)-Model_joints[7];
    float ratio1, ratio2;
    if(temp.at<float>(2,0)/Model_joints[8].at<float>(1,0)>1)
        ratio1 = 1;
    else
        ratio1 = temp.at<float>(2,0)/Model_joints[8].at<float>(1,0);

    double temp_alpha = asin(ratio1);
    if(sin(temp_alpha) == 1 ||temp.at<float>(0,0)/Model_joints[8].at<float>(1,0)/cos(temp_alpha)>1)
        ratio1 = 1;
    else
        ratio1 = temp.at<float>(0,0)/Model_joints[8].at<float>(1,0)/cos(temp_alpha);
    double temp_gama = asin(-ratio1);
    //    temp_alpha += acos(temp.at<float>(1,0)/Model_joints[8].at<float>(1,0)/cos(temp_gama));
    //    temp_alpha = temp_alpha/2.0;
    //    temp_gama += acos(temp.at<float>(1,0)/Model_joints[8].at<float>(1,0)/cos(temp_alpha));
    //    temp_gama = temp_gama/2.0;
    parameters[11] = arc2degree(temp_alpha);
    parameters[10] = arc2degree(temp_gama);

    R = R_z(parameters[10])*R_x(parameters[11]);

    for(int i = 8; i< 11; i++){
        joints_position_Mat.col(i) = R.inv()*(joints_position_Mat.col(i)-joints_position_Mat.col(7));
        //std::cout << joints_position_Mat.col(i) << std::endl;
    }

    //2.2 parameter 12, 13:
    temp = joints_position_Mat.col(9)-joints_position_Mat.col(8);

    if(temp.at<float>(2,0)/Model_joints[9].at<float>(1,0)>1)
        ratio1 = 1;
    else
        ratio1 = temp.at<float>(2,0)/Model_joints[9].at<float>(1,0);

    if(temp.at<float>(1,0)/Model_joints[9].at<float>(1,0)>1)
        ratio2 = 1;
    else
        ratio2 = temp.at<float>(1,0)/Model_joints[9].at<float>(1,0);

    temp_alpha = (asin(ratio1)+acos(ratio2))/2.0;

    parameters[12] = arc2degree(temp_alpha);

    for(int i =9; i<11; i++){
        joints_position_Mat.col(i) = R_x(parameters[12]).inv()*(joints_position_Mat.col(i)-joints_position_Mat.col(8));
    }

    temp = joints_position_Mat.col(10) - joints_position_Mat.col(9);
    if (temp.at<float>(2,0)/Model_joints[10].at<float>(1,0)>1)
        ratio1 = 1;
    else
        ratio1 = temp.at<float>(2,0)/Model_joints[10].at<float>(1,0);

    if (temp.at<float>(1,0)/Model_joints[10].at<float>(1,0)>1)
        ratio2 = 1;
    else
        ratio2 = temp.at<float>(1,0)/Model_joints[10].at<float>(1,0);
    temp_alpha = (asin(ratio1)+acos(ratio2))/2.0;

    parameters[13] = arc2degree(temp_alpha);

    //3. find angles of joints of middle, ring, pinky fingers:
    for( int finger = 0; finger < 3; finger++){
        temp = joints_position_Mat.col(13+5*finger)-Model_joints[12+5*finger];

        temp_alpha = asin(temp.at<float>(2,0)/Model_joints[13+5*finger].at<float>(1,0));
        temp_gama = asin(-temp.at<float>(0,0)/Model_joints[13+5*finger].at<float>(1,0)/cos(temp_alpha));
        //        temp_alpha += acos(temp.at<float>(1,0)/Model_joints[13+5*finger].at<float>(1,0)/cos(temp_gama));
        //        temp_alpha = temp_alpha/2.0;
        //        temp_gama += acos(temp.at<float>(1,0)/Model_joints[13+5*finger].at<float>(1,0)/cos(temp_alpha));
        //        temp_gama = temp_gama/2.0;

        parameters[15+finger*4] = arc2degree(temp_alpha);
        parameters[14+finger*4] = arc2degree(temp_gama);

        R = R_z(parameters[14+finger*4])*R_x(parameters[15+finger*4]);

        for(int i = 13+5*finger; i< 16+5*finger; i++){
            joints_position_Mat.col(i) = R.inv()*(joints_position_Mat.col(i)-joints_position_Mat.col(12+5*finger));
            //std::cout << joints_position_Mat.col(i) << std::endl;
        }


        temp = joints_position_Mat.col(14+5*finger)-joints_position_Mat.col(13+5*finger);

        if(temp.at<float>(2,0)/Model_joints[14+5*finger].at<float>(1,0)>1)
            ratio1 = 1;
        else
            ratio1 = temp.at<float>(2,0)/Model_joints[14+5*finger].at<float>(1,0);

        if(temp.at<float>(1,0)/Model_joints[14+5*finger].at<float>(1,0)>1){
            ratio2 = 1;
        }
        else
            ratio2 = temp.at<float>(1,0)/Model_joints[14+5*finger].at<float>(1,0);

        temp_alpha = (asin(ratio1)+acos(ratio2))/2.0;

        parameters[16+finger*4] = arc2degree(temp_alpha);


        for(int i = 14+5*finger; i<16+5*finger; i++){
            joints_position_Mat.col(i) = R_x(parameters[16+finger*4]).inv()*(joints_position_Mat.col(i)-joints_position_Mat.col(13+5*finger));
        }

        temp = joints_position_Mat.col(15+5*finger) - joints_position_Mat.col(14+5*finger);

        if(temp.at<float>(2,0)/Model_joints[15+5*finger].at<float>(1,0)>1)
            ratio1 = 1;
        else
            ratio1 = temp.at<float>(2,0)/Model_joints[15+5*finger].at<float>(1,0);

        if(temp.at<float>(1,0)/Model_joints[15+5*finger].at<float>(1,0)>1)
            ratio2 = 1;
        else
            ratio2 = temp.at<float>(1,0)/Model_joints[15+5*finger].at<float>(1,0);
        temp_alpha = (asin(ratio1)+acos(ratio2))/2.0;


        parameters[17+finger*4] = arc2degree(temp_alpha);
    }

    //4. find angles of joints of thumb
    temp = joints_position_Mat.col(2)-Model_joints[1];
    if(temp.at<float>(2,0)/Model_joints[2].at<float>(1,0)>1)
        ratio1 = 1;
    else
        ratio1 = temp.at<float>(2,0)/Model_joints[2].at<float>(1,0);

    temp_alpha = asin(ratio1);
    if(sin(temp_alpha) == 1 ||temp.at<float>(0,0)/Model_joints[2].at<float>(1,0)/cos(temp_alpha)>1)
        ratio1 = 1;
    else
        ratio1 = temp.at<float>(0,0)/Model_joints[2].at<float>(1,0)/cos(temp_alpha);
    temp_gama = asin(-ratio1);
    //    temp_alpha += acos(temp.at<float>(1,0)/Model_joints[8].at<float>(1,0)/cos(temp_gama));
    //    temp_alpha = temp_alpha/2.0;
    //    temp_gama += acos(temp.at<float>(1,0)/Model_joints[8].at<float>(1,0)/cos(temp_alpha));
    //    temp_gama = temp_gama/2.0;
    parameters[7] = arc2degree(temp_alpha);
    parameters[6] = arc2degree(temp_gama)-10;

    R = R_z(parameters[6]+10)*R_x(parameters[7])*R_y(70);

    for(int i = 2; i< 5; i++){
        joints_position_Mat.col(i) = R.inv()*(joints_position_Mat.col(i)-joints_position_Mat.col(1));
        //std::cout << joints_position_Mat.col(i) << std::endl;
    }

    //4.2 parameter 8, 9:
    temp = joints_position_Mat.col(3)-joints_position_Mat.col(2);

    if(temp.at<float>(2,0)/Model_joints[3].at<float>(1,0)>1)
        ratio1 = 1;
    else
        ratio1 = temp.at<float>(2,0)/Model_joints[3].at<float>(1,0);

    if(temp.at<float>(1,0)/Model_joints[3].at<float>(1,0)>1)
        ratio2 = 1;
    else
        ratio2 = temp.at<float>(1,0)/Model_joints[3].at<float>(1,0);

    temp_alpha = (asin(ratio1)+acos(ratio2))/2.0;

    parameters[8] = arc2degree(temp_alpha);

    for(int i =3; i<5; i++){
        joints_position_Mat.col(i) = R_x(parameters[8]).inv()*(joints_position_Mat.col(i)-joints_position_Mat.col(2));
    }

    temp = joints_position_Mat.col(4) - joints_position_Mat.col(3);
    if (temp.at<float>(2,0)/Model_joints[4].at<float>(1,0)>1)
        ratio1 = 1;
    else
        ratio1 = temp.at<float>(2,0)/Model_joints[4].at<float>(1,0);

    if (temp.at<float>(1,0)/Model_joints[4].at<float>(1,0)>1)
        ratio2 = 1;
    else
        ratio2 = temp.at<float>(1,0)/Model_joints[4].at<float>(1,0);

    temp_alpha = (asin(ratio1)+acos(ratio2))/2.0;

    parameters[9] = arc2degree(temp_alpha);

    for(int i = 0; i< 26; i++){
        std::cout << i << ": " << parameters[i]<<std::endl;
    }



}

void articulate_HandModel_XYZRGB::get_joints_positions(){

    Mat joints_for_calc[26];
    for(int i = 0; i < 26; i++){
        joints_for_calc[i] = Mat::zeros(3,1,CV_32FC1);
        Model_joints[i].copyTo(joints_for_calc[i]);
    }
    //1. palm joints: 1,6,11,16,21,7,12,17,22
    //palm joints with reference to palm/hand coordinate:
    //palm.thumb
    Model_joints[1].copyTo(joints_for_calc[1]);
    //palm.index
    Model_joints[6].copyTo(joints_for_calc[6]);
    Model_joints[7].copyTo(joints_for_calc[7]);
    //palm.middle
    Model_joints[11].copyTo(joints_for_calc[11]);
    Model_joints[12].copyTo(joints_for_calc[12]);
    //palm.ring
    Model_joints[16].copyTo(joints_for_calc[16]);
    Model_joints[17].copyTo(joints_for_calc[17]);
    //palm.pinky
    Model_joints[21].copyTo(joints_for_calc[21]);
    Model_joints[22].copyTo(joints_for_calc[22]);

    //2.fingers:
    //2.1 index(extrinsic):
    Mat R[3];
    R[0] = R_z(parameters[10])*R_x(parameters[11]);
    R[1] = R_x(parameters[12]);
    R[2] = R_x(parameters[13]);

    joints_for_calc[10] = R[0]*(R[1]*(R[2]*joints_for_calc[10]+joints_for_calc[9])+joints_for_calc[8])+joints_for_calc[7];
    joints_for_calc[9] = R[0]*(R[1]*joints_for_calc[9]+joints_for_calc[8])+joints_for_calc[7];
    joints_for_calc[8] = R[0]*joints_for_calc[8]+joints_for_calc[7];

    //2.2 middel to pinky(extrinsic):
    for ( int i = 0; i < 3; ++i){
        R[0] = R_z(parameters[i*4+14])*R_x(parameters[i*4+15]);
        R[1] = R_x(parameters[i*4+16]);
        R[2] = R_x(parameters[i*4+17]);

        joints_for_calc[i*5+15] = R[0]*(R[1]*(R[2]*joints_for_calc[i*5+15]+joints_for_calc[i*5+14])+joints_for_calc[i*5+13])+joints_for_calc[i*5+12];
        joints_for_calc[i*5+14] = R[0]*(R[1]*joints_for_calc[i*5+14]+joints_for_calc[i*5+13])+joints_for_calc[i*5+12];
        joints_for_calc[i*5+13] = R[0]*joints_for_calc[i*5+13]+joints_for_calc[i*5+12];

    }

    //2.3 thumb(extrinsic)
    R[0] = R_y(70);
    R[0] = R_z(10+parameters[6])*R_x(parameters[7])*R[0];
    R[1] = R_x(parameters[8]);
    R[2] = R_x(parameters[9]);

    //    cv::Mat mtxR, mtxQ;
    //    cv::Vec3d angles;
    //    angles = cv::RQDecomp3x3(R[0]*R_y(70).inv(), mtxR, mtxQ);
    //    std::cout<<"0: "<<angles[0]<<std::endl;
    //    std::cout<<"1: "<<angles[1]<<std::endl;
    //    std::cout<<"2: "<<angles[2]<<std::endl;

    joints_for_calc[4] = R[0]*(R[1]*(R[2]*joints_for_calc[4]+joints_for_calc[3])+joints_for_calc[2])+joints_for_calc[1];
    joints_for_calc[3] = R[0]*(R[1]*joints_for_calc[3]+joints_for_calc[2])+joints_for_calc[1];
    joints_for_calc[2] = R[0]*joints_for_calc[2]+joints_for_calc[1];
    joints_for_calc[5].at<float>(0,0) = joints_for_calc[4].at<float>(0,0);
    joints_for_calc[5].at<float>(1,0) = joints_for_calc[4].at<float>(1,0);
    joints_for_calc[5].at<float>(2,0) = joints_for_calc[4].at<float>(2,0);


    //3. palm after pitch yaw roll(extrinsic):
    Mat R_p_r_y = R_z(parameters[5])*R_y(parameters[4])*R_x(parameters[3]);
    //std::cout << "R: " << R_p_r_y << std::endl;
    Mat translation = Mat::zeros(3,1,CV_32FC1);
    translation.at<float>(0,0) = parameters[0];
    translation.at<float>(1,0) = parameters[1];
    translation.at<float>(2,0) = parameters[2];

    //std::cout << "translation: " << translation << std::endl;
    for(int i = 0; i< 26; i++){
        joints_for_calc[i] = R_p_r_y * joints_for_calc[i]+translation;
    }



    //4. put calculation results into joints_position
    for(int i = 0; i< 26; i++){
        joints_position[i].x = joints_for_calc[i].at<float>(0,0);
        joints_position[i].y = joints_for_calc[i].at<float>(0,1);
        joints_position[i].z = joints_for_calc[i].at<float>(0,2);
        //std::cout<< i <<": "<<joints_position[i]<<std::endl;
    }

}

void articulate_HandModel_XYZRGB::set_joints_positions(pcl::PointCloud<pcl::PointXYZRGB> hand_kp){

    if(hand_kp.size() == 32){
        joints_position[0] = hand_kp.points[0];
        joints_position[1].rgb = hand_kp.points[2].rgb;
        joints_position[1].x = (hand_kp.points[2].x+hand_kp.points[8].x)/2.0;
        joints_position[1].y = (hand_kp.points[2].y+hand_kp.points[8].y)/2.0;
        joints_position[1].z = (hand_kp.points[2].z+hand_kp.points[8].z)/2.0;
        joints_position[2] = hand_kp.points[4];
        joints_position[3] = hand_kp.points[5];
        joints_position[4] = hand_kp.points[6];
        joints_position[5] = joints_position[4];

        for(int i = 1; i< 5; i++){
            for(int j = 0; j < 5; j++){
                joints_position[i*5+1+j] = hand_kp.points[i*6+j+2];
            }
        }

    }
}

void articulate_HandModel_XYZRGB::CP_palm_fitting1(Mat Hand_DepthMat,Mat LabelMat, int resolution, vector<int> labelflag){
    //1. CP palm:
    //1.1 put palm joints model into discrete space
    int imageSize = int(LabelMat.rows);
    Mat palm;
    palm = Mat::zeros(3, 9, CV_32FC1);

    palm.at<float>(0,0) = 0;
    palm.at<float>(1,0) = 0;
    palm.at<float>(2,0) = 0;

    palm.at<float>(0,1) = joints_position[6].x;
    palm.at<float>(1,1) = joints_position[6].y;
    palm.at<float>(2,1) = joints_position[6].z;

    palm.at<float>(0,2) = joints_position[7].x;
    palm.at<float>(1,2) = joints_position[7].y;
    palm.at<float>(2,2) = joints_position[7].z;

    palm.at<float>(0,3) = joints_position[11].x;
    palm.at<float>(1,3) = joints_position[11].y;
    palm.at<float>(2,3) = joints_position[11].z;

    palm.at<float>(0,4) = joints_position[12].x;
    palm.at<float>(1,4) = joints_position[12].y;
    palm.at<float>(2,4) = joints_position[12].z;

    palm.at<float>(0,5) = joints_position[16].x;
    palm.at<float>(1,5) = joints_position[16].y;
    palm.at<float>(2,5) = joints_position[16].z;

    palm.at<float>(0,6) = joints_position[17].x;
    palm.at<float>(1,6) = joints_position[17].y;
    palm.at<float>(2,6) = joints_position[17].z;

    palm.at<float>(0,7) = joints_position[21].x;
    palm.at<float>(1,7) = joints_position[21].y;
    palm.at<float>(2,7) = joints_position[21].z;

    palm.at<float>(0,8) = joints_position[22].x;
    palm.at<float>(1,8) = joints_position[22].y;
    palm.at<float>(2,8) = joints_position[22].z;
    Mat palm_discrete = Mat::zeros(3, 9, CV_32FC1);
    palm_discrete = palm*1000/resolution+imageSize/2;

    //    Mat show = Mat::zeros(imageSize, imageSize, CV_32FC1);
    //    for(int i = 1; i< 9; i++){
    //        show.at<float>(int(palm_discrete.at<float>(1,i)), int(palm_discrete.at<float>(0,i))) = 255;
    //    }
    //    imshow("show", show);
    //    //cv::waitKey();

    //1.2 find nearest neighbour:
    double temp_dis[8] = {1000,1000,1000,1000, 1000, 1000,1000,1000};
    int temp_row[8], temp_col[8];
    for( int row = 0; row < LabelMat.rows; row++){
        for(int col = 0; col < LabelMat.cols; col++){
            switch(int(LabelMat.at<unsigned char>(row, col))){
            case 1:
            {

                double dis = (Hand_DepthMat.at<unsigned char>(row, col) - palm_discrete.at<float>(2,1)) * (Hand_DepthMat.at<unsigned char>(row, col) - palm_discrete.at<float>(2,1))+
                        (row - palm_discrete.at<float>(1,1)) * (row - palm_discrete.at<float>(1,1)) +
                        (col - palm_discrete.at<float>(0,1)) * (col - palm_discrete.at<float>(0,1));
                if (dis < temp_dis[0]){
                    temp_row[0] = row;
                    temp_col[0] = col;
                    temp_dis[0] = dis;
                }
                //                    std::cout << "dis1: " << Hand_DepthMat.at<unsigned char>(row, col) <<std::endl;
                //                    std::cout << "dis2: " << palm.at<float>(2,1) <<std::endl;
                //                    std::cout << "dis3: " << row <<std::endl;
                //                    std::cout << "dis4: " << palm.at<float>(0,1) <<std::endl;
                //                    std::cout << "dis5: " << col <<std::endl;
                //                    std::cout << "dis6: " << palm.at<float>(1,1) <<std::endl;

                dis = (Hand_DepthMat.at<unsigned char>(row, col) - palm_discrete.at<float>(2,2)) * (Hand_DepthMat.at<unsigned char>(row, col) - palm_discrete.at<float>(2,2))+
                        (row - palm_discrete.at<float>(1,2)) * (row - palm_discrete.at<float>(1,2)) +
                        (col - palm_discrete.at<float>(0,2)) * (col - palm_discrete.at<float>(0,2));
                if (dis < temp_dis[1]){
                    temp_row[1] = row;
                    temp_col[1] = col;
                    temp_dis[1] = dis;
                }

                break;
            }

            case 2:
            {
                double dis = (Hand_DepthMat.at<unsigned char>(row, col) - palm_discrete.at<float>(2,3)) * (Hand_DepthMat.at<unsigned char>(row, col) - palm_discrete.at<float>(2,3))+
                        (row - palm_discrete.at<float>(1,3)) * (row - palm_discrete.at<float>(1,3)) +
                        (col - palm_discrete.at<float>(0,3)) * (col - palm_discrete.at<float>(0,3));
                if (dis < temp_dis[2]){
                    temp_row[2] = row;
                    temp_col[2] = col;
                    temp_dis[2] = dis;
                }
                dis = (Hand_DepthMat.at<unsigned char>(row, col) - palm_discrete.at<float>(2,4)) * (Hand_DepthMat.at<unsigned char>(row, col) - palm_discrete.at<float>(2,4))+
                        (row - palm_discrete.at<float>(1,4)) * (row - palm_discrete.at<float>(1,4)) +
                        (col - palm_discrete.at<float>(0,4)) * (col - palm_discrete.at<float>(0,4));
                if (dis < temp_dis[3]){
                    temp_row[3] = row;
                    temp_col[3] = col;
                    temp_dis[3] = dis;
                }
                break;
            }

            case 3:
            {
                double dis = (Hand_DepthMat.at<unsigned char>(row, col) - palm_discrete.at<float>(2,5)) * (Hand_DepthMat.at<unsigned char>(row, col) - palm_discrete.at<float>(2,5))+
                        (row - palm_discrete.at<float>(1,5)) * (row - palm_discrete.at<float>(1,5)) +
                        (col - palm_discrete.at<float>(0,5)) * (col - palm_discrete.at<float>(0,5));
                if (dis < temp_dis[4]){
                    temp_row[4] = row;
                    temp_col[4] = col;
                    temp_dis[4] = dis;
                }
                dis = (Hand_DepthMat.at<unsigned char>(row, col) - palm_discrete.at<float>(2,6)) * (Hand_DepthMat.at<unsigned char>(row, col) - palm_discrete.at<float>(2,6))+
                        (row - palm_discrete.at<float>(1,6)) * (row - palm_discrete.at<float>(1,6)) +
                        (col - palm_discrete.at<float>(0,6)) * (col - palm_discrete.at<float>(0,6));
                if (dis < temp_dis[5]){
                    temp_row[5] = row;
                    temp_col[5] = col;
                    temp_dis[5] = dis;
                }
                break;
            }

            case 4:
            {
                double dis = (Hand_DepthMat.at<unsigned char>(row, col) - palm_discrete.at<float>(2,7)) * (Hand_DepthMat.at<unsigned char>(row, col) - palm_discrete.at<float>(2,7))+
                        (row - palm_discrete.at<float>(1,7)) * (row - palm_discrete.at<float>(1,7)) +
                        (col - palm_discrete.at<float>(0,7)) * (col - palm_discrete.at<float>(0,7));
                if (dis < temp_dis[6]){
                    temp_row[6] = row;
                    temp_col[6] = col;
                    temp_dis[6] = dis;
                }
                dis = (Hand_DepthMat.at<unsigned char>(row, col) - palm_discrete.at<float>(2,8)) * (Hand_DepthMat.at<unsigned char>(row, col) - palm_discrete.at<float>(2,8))+
                        (row - palm_discrete.at<float>(1,8)) * (row - palm_discrete.at<float>(1,8)) +
                        (col - palm_discrete.at<float>(0,8)) * (col - palm_discrete.at<float>(0,8));
                if (dis < temp_dis[7]){
                    temp_row[7] = row;
                    temp_col[7] = col;
                    temp_dis[7] = dis;
                }
            }

            default:
            {
                ;
            }

            }

        }
    }
    //    for(int i = 0; i<8; i++){
    //        std::cout <<i<<": " << temp_row[i] <<" " << temp_col[i]<< " " << temp_dis[i]<<std::endl;
    //    }

    //        Mat show2 = Mat::zeros(imageSize, imageSize, CV_32FC1);
    //        for(int i = 0; i< 8; i++){
    //            show2.at<float>(temp_row[i], temp_col[i]) = 255;
    //        }
    //        imshow("show2", show2);
    //        imshow("label", LabelMat*15);
    //        cv::waitKey();

    //1.3 estimate palm pose:
    //1.3.1 put nearest neighbour back to 3D space:
    Mat Oberservation = Mat::zeros(3, 9, CV_32FC1);
    for(int i = 1; i < 9; i++){
        Oberservation.at<float>(0,i) = (temp_col[i-1]-imageSize/2.0)*resolution/1000.0;
        Oberservation.at<float>(1,i) = (temp_row[i-1]-imageSize/2.0)*resolution/1000.0;
        Oberservation.at<float>(2,i) = (Hand_DepthMat.at<unsigned char>(temp_row[i-1], temp_col[i-1])-imageSize/2.0)*resolution/1000.0;
    }

    //1.3.2 compute
    Mat R,t;
    t = Mat::zeros(3, 1, CV_32FC1);
    Mat A(Oberservation, Rect(1,0,8,3)), B(palm_model, Rect(0,0,8,3));

    poseEstimate::poseestimate::compute(A,B,R,t);

    //    std::cout<<"Ober: "<< Oberservation << std::endl;
    //    std::cout << "palm: " << palm << std::endl;

    //    std::cout<<"R: "<< R << std::endl;
    std::cout << "t: " << t << std::endl;
    //1.3.3 get the angles:
    cv::Mat mtxR, mtxQ;
    cv::Vec3d angles;
    angles = cv::RQDecomp3x3(R, mtxR, mtxQ);
    std::cout<<"angles: " << angles <<std::endl;

    parameters[3] = angles[0];
    parameters[4] = angles[1];
    parameters[5] = angles[2];

    //    parameters[0] = t.at<float>(0,0);
    //    parameters[1] = t.at<float>(1,0);
    parameters[2] = t.at<float>(2,0);

}

void articulate_HandModel_XYZRGB::CP_palm_fitting2(Mat Hand_DepthMat,Mat LabelMat, int resolution, vector<int> labelflag){
    //1. CP palm:
    //1.1 put palm joints model into discrete space
    int imageSize = int(LabelMat.rows);
    Mat palm;
    palm = Mat::zeros(3, 10, CV_32FC1);

    palm.at<float>(0,0) = 0;
    palm.at<float>(1,0) = 0;
    palm.at<float>(2,0) = 0;

    palm.at<float>(0,1) = joints_position[6].x;
    palm.at<float>(1,1) = joints_position[6].y;
    palm.at<float>(2,1) = joints_position[6].z;

    palm.at<float>(0,2) = joints_position[7].x;
    palm.at<float>(1,2) = joints_position[7].y;
    palm.at<float>(2,2) = joints_position[7].z;

    palm.at<float>(0,3) = joints_position[11].x;
    palm.at<float>(1,3) = joints_position[11].y;
    palm.at<float>(2,3) = joints_position[11].z;

    palm.at<float>(0,4) = joints_position[12].x;
    palm.at<float>(1,4) = joints_position[12].y;
    palm.at<float>(2,4) = joints_position[12].z;

    palm.at<float>(0,5) = joints_position[16].x;
    palm.at<float>(1,5) = joints_position[16].y;
    palm.at<float>(2,5) = joints_position[16].z;

    palm.at<float>(0,6) = joints_position[17].x;
    palm.at<float>(1,6) = joints_position[17].y;
    palm.at<float>(2,6) = joints_position[17].z;

    palm.at<float>(0,7) = joints_position[21].x;
    palm.at<float>(1,7) = joints_position[21].y;
    palm.at<float>(2,7) = joints_position[21].z;

    palm.at<float>(0,8) = joints_position[22].x;
    palm.at<float>(1,8) = joints_position[22].y;
    palm.at<float>(2,8) = joints_position[22].z;

    palm.at<float>(0,9) = joints_position[1].x;
    palm.at<float>(1,9) = joints_position[1].y;
    palm.at<float>(2,9) = joints_position[1].z;
    Mat palm_discrete = Mat::zeros(3, 10, CV_32FC1);
    palm_discrete = palm*1000/resolution+imageSize/2;

    //    Mat show = Mat::zeros(imageSize, imageSize, CV_32FC1);
    //    for(int i = 1; i< 9; i++){
    //        show.at<float>(int(palm_discrete.at<float>(1,i)), int(palm_discrete.at<float>(0,i))) = 255;
    //    }
    //    imshow("show", show);
    //    //cv::waitKey();

    //1.2 find nearest neighbour:
    double temp_dis[9] = {1000,1000,1000,1000, 1000, 1000,1000,1000, 1000};
    int temp_row[9], temp_col[9];
    for( int row = 0; row < LabelMat.rows; row++){
        for(int col = 0; col < LabelMat.cols; col++){
            switch(int(LabelMat.at<unsigned char>(row, col))){
            case 1:
            {

                double dis = (Hand_DepthMat.at<unsigned char>(row, col) - palm_discrete.at<float>(2,1)) * (Hand_DepthMat.at<unsigned char>(row, col) - palm_discrete.at<float>(2,1))+
                        (row - palm_discrete.at<float>(1,1)) * (row - palm_discrete.at<float>(1,1)) +
                        (col - palm_discrete.at<float>(0,1)) * (col - palm_discrete.at<float>(0,1));
                if (dis < temp_dis[0]){
                    temp_row[0] = row;
                    temp_col[0] = col;
                    temp_dis[0] = dis;
                }
                //                    std::cout << "dis1: " << Hand_DepthMat.at<unsigned char>(row, col) <<std::endl;
                //                    std::cout << "dis2: " << palm.at<float>(2,1) <<std::endl;
                //                    std::cout << "dis3: " << row <<std::endl;
                //                    std::cout << "dis4: " << palm.at<float>(0,1) <<std::endl;
                //                    std::cout << "dis5: " << col <<std::endl;
                //                    std::cout << "dis6: " << palm.at<float>(1,1) <<std::endl;

                dis = (Hand_DepthMat.at<unsigned char>(row, col) - palm_discrete.at<float>(2,2)) * (Hand_DepthMat.at<unsigned char>(row, col) - palm_discrete.at<float>(2,2))+
                        (row - palm_discrete.at<float>(1,2)) * (row - palm_discrete.at<float>(1,2)) +
                        (col - palm_discrete.at<float>(0,2)) * (col - palm_discrete.at<float>(0,2));
                if (dis < temp_dis[1]){
                    temp_row[1] = row;
                    temp_col[1] = col;
                    temp_dis[1] = dis;
                }

                break;
            }

            case 2:
            {
                double dis = (Hand_DepthMat.at<unsigned char>(row, col) - palm_discrete.at<float>(2,3)) * (Hand_DepthMat.at<unsigned char>(row, col) - palm_discrete.at<float>(2,3))+
                        (row - palm_discrete.at<float>(1,3)) * (row - palm_discrete.at<float>(1,3)) +
                        (col - palm_discrete.at<float>(0,3)) * (col - palm_discrete.at<float>(0,3));
                if (dis < temp_dis[2]){
                    temp_row[2] = row;
                    temp_col[2] = col;
                    temp_dis[2] = dis;
                }
                dis = (Hand_DepthMat.at<unsigned char>(row, col) - palm_discrete.at<float>(2,4)) * (Hand_DepthMat.at<unsigned char>(row, col) - palm_discrete.at<float>(2,4))+
                        (row - palm_discrete.at<float>(1,4)) * (row - palm_discrete.at<float>(1,4)) +
                        (col - palm_discrete.at<float>(0,4)) * (col - palm_discrete.at<float>(0,4));
                if (dis < temp_dis[3]){
                    temp_row[3] = row;
                    temp_col[3] = col;
                    temp_dis[3] = dis;
                }
                break;
            }

            case 3:
            {
                double dis = (Hand_DepthMat.at<unsigned char>(row, col) - palm_discrete.at<float>(2,5)) * (Hand_DepthMat.at<unsigned char>(row, col) - palm_discrete.at<float>(2,5))+
                        (row - palm_discrete.at<float>(1,5)) * (row - palm_discrete.at<float>(1,5)) +
                        (col - palm_discrete.at<float>(0,5)) * (col - palm_discrete.at<float>(0,5));
                if (dis < temp_dis[4]){
                    temp_row[4] = row;
                    temp_col[4] = col;
                    temp_dis[4] = dis;
                }
                dis = (Hand_DepthMat.at<unsigned char>(row, col) - palm_discrete.at<float>(2,6)) * (Hand_DepthMat.at<unsigned char>(row, col) - palm_discrete.at<float>(2,6))+
                        (row - palm_discrete.at<float>(1,6)) * (row - palm_discrete.at<float>(1,6)) +
                        (col - palm_discrete.at<float>(0,6)) * (col - palm_discrete.at<float>(0,6));
                if (dis < temp_dis[5]){
                    temp_row[5] = row;
                    temp_col[5] = col;
                    temp_dis[5] = dis;
                }
                break;
            }

            case 4:
            {
                double dis = (Hand_DepthMat.at<unsigned char>(row, col) - palm_discrete.at<float>(2,7)) * (Hand_DepthMat.at<unsigned char>(row, col) - palm_discrete.at<float>(2,7))+
                        (row - palm_discrete.at<float>(1,7)) * (row - palm_discrete.at<float>(1,7)) +
                        (col - palm_discrete.at<float>(0,7)) * (col - palm_discrete.at<float>(0,7));
                if (dis < temp_dis[6]){
                    temp_row[6] = row;
                    temp_col[6] = col;
                    temp_dis[6] = dis;
                }
                dis = (Hand_DepthMat.at<unsigned char>(row, col) - palm_discrete.at<float>(2,8)) * (Hand_DepthMat.at<unsigned char>(row, col) - palm_discrete.at<float>(2,8))+
                        (row - palm_discrete.at<float>(1,8)) * (row - palm_discrete.at<float>(1,8)) +
                        (col - palm_discrete.at<float>(0,8)) * (col - palm_discrete.at<float>(0,8));
                if (dis < temp_dis[7]){
                    temp_row[7] = row;
                    temp_col[7] = col;
                    temp_dis[7] = dis;
                }
                break;
            }

            case 5:
            {
                double dis = (Hand_DepthMat.at<unsigned char>(row, col) - palm_discrete.at<float>(2,9)) * (Hand_DepthMat.at<unsigned char>(row, col) - palm_discrete.at<float>(2,9))+
                        (row - palm_discrete.at<float>(1,9)) * (row - palm_discrete.at<float>(1,9)) +
                        (col - palm_discrete.at<float>(0,9)) * (col - palm_discrete.at<float>(0,9));
                if (dis < temp_dis[8]){
                    temp_row[8] = row;
                    temp_col[8] = col;
                    temp_dis[8] = dis;
                }
            }

            default:
            {
                ;
            }

            }

        }
    }
    //    for(int i = 0; i<8; i++){
    //        std::cout <<i<<": " << temp_row[i] <<" " << temp_col[i]<< " " << temp_dis[i]<<std::endl;
    //    }

    //        Mat show2 = Mat::zeros(imageSize, imageSize, CV_32FC1);
    //        for(int i = 0; i< 8; i++){
    //            show2.at<float>(temp_row[i], temp_col[i]) = 255;
    //        }
    //        imshow("show2", show2);
    //        imshow("label", LabelMat*15);
    //        cv::waitKey();

    //1.3 estimate palm pose:
    //1.3.1 put nearest neighbour back to 3D space:
    Mat Oberservation = Mat::zeros(3, 10, CV_32FC1);
    for(int i = 1; i < 10; i++){
        Oberservation.at<float>(0,i) = (temp_col[i-1]-imageSize/2.0)*resolution/1000.0;
        Oberservation.at<float>(1,i) = (temp_row[i-1]-imageSize/2.0)*resolution/1000.0;
        Oberservation.at<float>(2,i) = (Hand_DepthMat.at<unsigned char>(temp_row[i-1], temp_col[i-1])-imageSize/2.0)*resolution/1000.0;
    }

    //1.3.2 compute
    Mat R,t;
    t = Mat::zeros(3, 1, CV_32FC1);
    Mat A(Oberservation, Rect(1,0,9,3)), B(palm_model, Rect(0,0,9,3));

    poseEstimate::poseestimate::compute(A,B,R,t);

    //    std::cout<<"Ober: "<< Oberservation << std::endl;
    //    std::cout << "palm: " << palm << std::endl;

    //    std::cout<<"R: "<< R << std::endl;
    std::cout << "t: " << t << std::endl;
    //1.3.3 get the angles:
    cv::Mat mtxR, mtxQ;
    cv::Vec3d angles;
    angles = cv::RQDecomp3x3(R, mtxR, mtxQ);
    std::cout<<"angles: " << angles <<std::endl;

    parameters[3] = angles[0];
    parameters[4] = angles[1];
    parameters[5] = angles[2];

    //    parameters[0] = t.at<float>(0,0);
    //    parameters[1] = t.at<float>(1,0);
    parameters[2] = t.at<float>(2,0);

}

void articulate_HandModel_XYZRGB::CP_palm_fitting3(Mat Hand_DepthMat,Mat LabelMat, int resolution, vector<int> labelflag, bool &ok_flag){

    vector< vector <Point3d> > Intersection(9, vector<Point3d>() );
    int imageSize = 300/resolution;
    //1. Looking for intersection region
    for(int row = 0; row < LabelMat.rows; row++){
        for(int col = 0; col < LabelMat.cols; col++){

            if( 0 < LabelMat.at<unsigned char>(row, col) && LabelMat.at<unsigned char>(row, col) < 6){
                //hand as well as palm edge:
                if(LabelMat.at<unsigned char>(row +1, col) == 0 || LabelMat.at<unsigned char>(row -1, col) == 0 || LabelMat.at<unsigned char>(row, col +1) == 0 || LabelMat.at<unsigned char>(row, col -1) == 0){
                    int L = LabelMat.at<unsigned char>(row, col);
                    Point3d p3d;
                    p3d.x = (col-imageSize/2.0)*resolution/1000.0;
                    p3d.y = (row-imageSize/2.0)*resolution/1000.0;
                    p3d.z = (Hand_DepthMat.at<unsigned char>(row, col)-imageSize/2.0)*resolution/1000.0;
                    Intersection[L+3].push_back(p3d);
                }
                //boundary between fingers and palm
                else if (LabelMat.at<unsigned char>(row, col) != 5 &&
                         3*LabelMat.at<unsigned char>(row, col) - LabelMat.at<unsigned char>(row +1, col) == -5 || 3*LabelMat.at<unsigned char>(row, col) - LabelMat.at<unsigned char>(row-1, col) == -5||
                         3*LabelMat.at<unsigned char>(row, col) - LabelMat.at<unsigned char>(row, col +1) == -5 || 3*LabelMat.at<unsigned char>(row, col) - LabelMat.at<unsigned char>(row, col-1) == -5){
                    int L = LabelMat.at<unsigned char>(row, col);
                    Point3d p3d;
                    p3d.x = (col-imageSize/2.0)*resolution/1000.0;
                    p3d.y = (row-imageSize/2.0)*resolution/1000.0;
                    p3d.z = (Hand_DepthMat.at<unsigned char>(row, col)-imageSize/2.0)*resolution/1000.0;
                    Intersection[L-1].push_back(p3d);
                }
            }
        }

    }
    for(int i = 0; i < 9; i++)
        std::cout<< "Size of " << i << ": " << Intersection[i].size() << std::endl;
    //1.2 Ransac to find the intersection center:
    Point3d center[9];
    int valid_number = 0;
    for(int i = 0; i < 7; i++){
        if(Intersection[i].size()!=0){
            Ransac(Intersection[i], center[i], 5, 0.02);
            valid_number++;
        }
        else
            center[i].x = -999;
        //std::cout<< "Center " << i << ": " << center[i] << std::endl;
    }

    if (valid_number > 4){
        ok_flag = true;

        //put the centers into oberservation:
        Mat Oberservation = Mat::zeros(3, valid_number, CV_32FC1);
        Mat palm = Mat::zeros(3, valid_number, CV_32FC1);;
        for(int i = 0; i < 7; i++){
            if(center[i].x != -999){
                Oberservation.at<float>(0,i) = center[i].x;
                Oberservation.at<float>(1,i) = center[i].y;
                Oberservation.at<float>(2,i) = center[i].z;
                int index;
                if(i < 4)
                    index = 5*i+7;
                else
                    index = 5*i-14;
                Model_joints[index].copyTo(palm.col(i));

            }
        }

        //1.3.2 compute
        Mat R,t;
        t = Mat::zeros(3, 1, CV_32FC1);

        poseEstimate::poseestimate::compute(Oberservation,palm,R,t);

        //    std::cout<<"Ober: "<< Oberservation << std::endl;
        //    std::cout << "palm: " << palm << std::endl;

        //    std::cout<<"R: "<< R << std::endl;
        std::cout << "t: " << t << std::endl;
        //1.3.3 get the angles:
        cv::Mat mtxR, mtxQ;
        cv::Vec3d angles;
        angles = cv::RQDecomp3x3(R, mtxR, mtxQ);
        std::cout<<"angles: " << angles <<std::endl;

        parameters[3] = angles[0];
        parameters[4] = angles[1];
        parameters[5] = angles[2];

        //    parameters[0] = t.at<float>(0,0);
        //    parameters[1] = t.at<float>(1,0);
        parameters[2] = t.at<float>(2,0);
    }
    else{
        ok_flag = false;
    }

}

void articulate_HandModel_XYZRGB::finger_fitting(Mat Hand_DepthMat,Mat LabelMat, int resolution, int bone){
    int imageSize = int(LabelMat.rows);
    cv::Point3d direction[5], temp_direction;
    double count[5] = {0,0,0,0,0};
    double length;

    for( int row = 0; row < LabelMat.rows; row++){
        for(int col = 0; col < LabelMat.cols; col++){
            switch(int(LabelMat.at<unsigned char>(row, col))-bone){
            //thumb proximal
            case 5:
            {
                temp_direction.z = (Hand_DepthMat.at<unsigned char>(row, col)-imageSize/2.0)*resolution/1000.0 - joints_position[1+bone].z;
                temp_direction.x = (col-imageSize/2.0)*resolution/1000.0 - joints_position[1+bone].x;
                temp_direction.y = (row-imageSize/2.0)*resolution/1000.0 - joints_position[1+bone].y;

                length = sqrt(temp_direction.x*temp_direction.x+temp_direction.y*temp_direction.y+temp_direction.z*temp_direction.z);
                //                count[0]+=length;
                direction[0].x += temp_direction.x*length;
                direction[0].y += temp_direction.y*length;
                direction[0].z += temp_direction.z*length;

                break;
            }

                //index finger proximal
            case 8:
            {
                temp_direction.z = (Hand_DepthMat.at<unsigned char>(row, col)-imageSize/2.0)*resolution/1000.0 - joints_position[7+bone].z;
                temp_direction.x = (col-imageSize/2.0)*resolution/1000.0 - joints_position[7+bone].x;
                temp_direction.y = (row-imageSize/2.0)*resolution/1000.0 - joints_position[7+bone].y;

                length = sqrt(temp_direction.x*temp_direction.x+temp_direction.y*temp_direction.y+temp_direction.z*temp_direction.z);
                //                                count[1]+=length;
                direction[1].x += temp_direction.x*length;
                direction[1].y += temp_direction.y*length;
                direction[1].z += temp_direction.z*length;

                break;
            }

                //middle finger proximal
            case 11:
            {
                temp_direction.z = (Hand_DepthMat.at<unsigned char>(row, col)-imageSize/2.0)*resolution/1000.0 - joints_position[12+bone].z;
                temp_direction.x = (col-imageSize/2.0)*resolution/1000.0 - joints_position[12+bone].x;
                temp_direction.y = (row-imageSize/2.0)*resolution/1000.0 - joints_position[12+bone].y;

                length = sqrt(temp_direction.x*temp_direction.x+temp_direction.y*temp_direction.y+temp_direction.z*temp_direction.z);
                //                count[2]+=length;
                direction[2].x += temp_direction.x*length;
                direction[2].y += temp_direction.y*length;
                direction[2].z += temp_direction.z*length;

                break;
            }

                //ring finger proximal
            case 14:
            {
                temp_direction.z = (Hand_DepthMat.at<unsigned char>(row, col)-imageSize/2.0)*resolution/1000.0 - joints_position[17+bone].z;
                temp_direction.x = (col-imageSize/2.0)*resolution/1000.0 - joints_position[17+bone].x;
                temp_direction.y = (row-imageSize/2.0)*resolution/1000.0 - joints_position[17+bone].y;

                length = sqrt(temp_direction.x*temp_direction.x+temp_direction.y*temp_direction.y+temp_direction.z*temp_direction.z);
                //                count[3]+=length;
                direction[3].x += temp_direction.x*length;
                direction[3].y += temp_direction.y*length;
                direction[3].z += temp_direction.z*length;

                break;
            }

                //little finger proximal
            case 17:
            {
                temp_direction.z = (Hand_DepthMat.at<unsigned char>(row, col)-imageSize/2.0)*resolution/1000.0 - joints_position[22+bone].z;
                temp_direction.x = (col-imageSize/2.0)*resolution/1000.0 - joints_position[22+bone].x;
                temp_direction.y = (row-imageSize/2.0)*resolution/1000.0 - joints_position[22+bone].y;

                length = sqrt(temp_direction.x*temp_direction.x+temp_direction.y*temp_direction.y+temp_direction.z*temp_direction.z);
                //                count[4]+=length;
                direction[4].x += temp_direction.x*length;
                direction[4].y += temp_direction.y*length;
                direction[4].z += temp_direction.z*length;

            }

            default:
            {
                ;
            }

            }
        }
    }

    for(int i = 0; i< 5; i++){
        length = sqrt(direction[i].x*direction[i].x+direction[i].y*direction[i].y+direction[i].z*direction[i].z);
        direction[i].x = direction[i].x/length;
        direction[i].y = direction[i].y/length;
        direction[i].z = direction[i].z/length;
    }

    joints_position[2+bone].x = joints_position[1+bone].x + bone_length[0][0+bone]/1000.0*direction[0].x;
    joints_position[2+bone].y = joints_position[1+bone].y + bone_length[0][0+bone]/1000.0*direction[0].y;
    joints_position[2+bone].z = joints_position[1+bone].z + bone_length[0][0+bone]/1000.0*direction[0].z;

    joints_position[8+bone].x = joints_position[7+bone].x + bone_length[1][1+bone]/1000.0*direction[1].x;
    joints_position[8+bone].y = joints_position[7+bone].y + bone_length[1][1+bone]/1000.0*direction[1].y;
    joints_position[8+bone].z = joints_position[7+bone].z + bone_length[1][1+bone]/1000.0*direction[1].z;

    joints_position[13+bone].x = joints_position[12+bone].x + bone_length[2][1+bone]/1000.0*direction[2].x;
    joints_position[13+bone].y = joints_position[12+bone].y + bone_length[2][1+bone]/1000.0*direction[2].y;
    joints_position[13+bone].z = joints_position[12+bone].z + bone_length[2][1+bone]/1000.0*direction[2].z;

    joints_position[18+bone].x = joints_position[17+bone].x + bone_length[3][1+bone]/1000.0*direction[3].x;
    joints_position[18+bone].y = joints_position[17+bone].y + bone_length[3][1+bone]/1000.0*direction[3].y;
    joints_position[18+bone].z = joints_position[17+bone].z + bone_length[3][1+bone]/1000.0*direction[3].z;

    joints_position[23+bone].x = joints_position[22+bone].x + bone_length[4][1+bone]/1000.0*direction[4].x;
    joints_position[23+bone].y = joints_position[22+bone].y + bone_length[4][1+bone]/1000.0*direction[4].y;
    joints_position[23+bone].z = joints_position[22+bone].z + bone_length[4][1+bone]/1000.0*direction[4].z;

    joints_position[5].x = joints_position[4].x;
    joints_position[5].y = joints_position[4].y;
    joints_position[5].z = joints_position[4].z;

}

void articulate_HandModel_XYZRGB::finger_fitting2(Mat Hand_DepthMat, Mat LabelMat, int resolution, int bone){

    int imageSize = int(LabelMat.rows);
    //1. CP finger:
    //1.1 put finger joints into discrete space and build mesh, put it into discrete space
    Mat finger_bone[5];
    //iterpolation
    //thumb:
    finger_bone[0] = Mat::zeros(3, 6, CV_32FC1);

    finger_bone[0].at<float>(0,0) = joints_position[1+bone].x;
    finger_bone[0].at<float>(1,0) = joints_position[1+bone].y;
    finger_bone[0].at<float>(2,0) = joints_position[1+bone].z;

    finger_bone[0].at<float>(0,1) = joints_position[2+bone].x;
    finger_bone[0].at<float>(1,1) = joints_position[2+bone].y;
    finger_bone[0].at<float>(2,1) = joints_position[2+bone].z;

    finger_bone[0].at<float>(0,2) = 1.0/5*joints_position[1+bone].x + 4.0/5*joints_position[2+bone].x;
    finger_bone[0].at<float>(1,2) = 1.0/5*joints_position[1+bone].y + 4.0/5*joints_position[2+bone].y;
    finger_bone[0].at<float>(2,2) = 1.0/5*joints_position[1+bone].z + 4.0/5*joints_position[2+bone].z;

    finger_bone[0].at<float>(0,3) = 2.0/5*joints_position[1+bone].x + 3.0/5*joints_position[2+bone].x;
    finger_bone[0].at<float>(1,3) = 2.0/5*joints_position[1+bone].y + 3.0/5*joints_position[2+bone].y;
    finger_bone[0].at<float>(2,3) = 2.0/5*joints_position[1+bone].z + 3.0/5*joints_position[2+bone].z;

    finger_bone[0].at<float>(0,4) = 3.0/5*joints_position[1+bone].x + 2.0/5*joints_position[2+bone].x;
    finger_bone[0].at<float>(1,4) = 3.0/5*joints_position[1+bone].y + 2.0/5*joints_position[2+bone].y;
    finger_bone[0].at<float>(2,4) = 3.0/5*joints_position[1+bone].z + 2.0/5*joints_position[2+bone].z;

    finger_bone[0].at<float>(0,5) = 4.0/5*joints_position[1+bone].x + 1.0/5*joints_position[2+bone].x;
    finger_bone[0].at<float>(1,5) = 4.0/5*joints_position[1+bone].y + 1.0/5*joints_position[2+bone].y;
    finger_bone[0].at<float>(2,5) = 4.0/5*joints_position[1+bone].z + 1.0/5*joints_position[2+bone].z;

    //other fingers
    for(int i = 1; i< 5; i++){
        finger_bone[i] = Mat::zeros(3, 6, CV_32FC1);

        finger_bone[i].at<float>(0,0) = joints_position[i*5+bone+2].x;
        finger_bone[i].at<float>(1,0) = joints_position[i*5+bone+2].y;
        finger_bone[i].at<float>(2,0) = joints_position[i*5+bone+2].z;

        finger_bone[i].at<float>(0,1) = joints_position[i*5+bone+3].x;
        finger_bone[i].at<float>(1,1) = joints_position[i*5+bone+3].y;
        finger_bone[i].at<float>(2,1) = joints_position[i*5+bone+3].z;

        finger_bone[i].at<float>(0,2) = 1.0/5*joints_position[i*5+bone+2].x + 4.0/5*joints_position[i*5+bone+3].x;
        finger_bone[i].at<float>(1,2) = 1.0/5*joints_position[i*5+bone+2].y + 4.0/5*joints_position[i*5+bone+3].y;
        finger_bone[i].at<float>(2,2) = 1.0/5*joints_position[i*5+bone+2].z + 4.0/5*joints_position[i*5+bone+3].z;

        finger_bone[i].at<float>(0,3) = 2.0/5*joints_position[i*5+bone+2].x + 3.0/5*joints_position[i*5+bone+3].x;
        finger_bone[i].at<float>(1,3) = 2.0/5*joints_position[i*5+bone+2].y + 3.0/5*joints_position[i*5+bone+3].y;
        finger_bone[i].at<float>(2,3) = 2.0/5*joints_position[i*5+bone+2].z + 3.0/5*joints_position[i*5+bone+3].z;

        finger_bone[i].at<float>(0,4) = 3.0/5*joints_position[i*5+bone+2].x + 2.0/5*joints_position[i*5+bone+3].x;
        finger_bone[i].at<float>(1,4) = 3.0/5*joints_position[i*5+bone+2].y + 2.0/5*joints_position[i*5+bone+3].y;
        finger_bone[i].at<float>(2,4) = 3.0/5*joints_position[i*5+bone+2].z + 2.0/5*joints_position[i*5+bone+3].z;

        finger_bone[i].at<float>(0,5) = 4.0/5*joints_position[i*5+bone+2].x + 1.0/5*joints_position[i*5+bone+3].x;
        finger_bone[i].at<float>(1,5) = 4.0/5*joints_position[i*5+bone+2].y + 1.0/5*joints_position[i*5+bone+3].y;
        finger_bone[i].at<float>(2,5) = 4.0/5*joints_position[i*5+bone+2].z + 1.0/5*joints_position[i*5+bone+3].z;
    }

    //    //thumb:
    //    finger_bone[0] = Mat::zeros(3, 6, CV_32FC1);

    //    finger_bone[0].at<float>(0,0) = joints_position[1+bone].x;
    //    finger_bone[0].at<float>(1,0) = joints_position[1+bone].y;
    //    finger_bone[0].at<float>(2,0) = joints_position[1+bone].z;

    //    finger_bone[0].at<float>(0,1) = joints_position[2+bone].x;
    //    finger_bone[0].at<float>(1,1) = joints_position[2+bone].y;
    //    finger_bone[0].at<float>(2,1) = joints_position[2+bone].z;

    //    finger_bone[0].at<float>(0,2) = 1.0/3*joints_position[1+bone].x + 2.0/3*joints_position[2+bone].x;
    //    finger_bone[0].at<float>(1,2) = joints_position[1+bone].y;
    //    finger_bone[0].at<float>(2,2) = 1.0/2*joints_position[1+bone].z + 1.0/2*joints_position[2+bone].z;

    //    finger_bone[0].at<float>(0,3) = 2.0/3*joints_position[1+bone].x + 1.0/3*joints_position[2+bone].x;
    //    finger_bone[0].at<float>(1,3) = joints_position[1+bone].y;
    //    finger_bone[0].at<float>(2,3) = 1.0/2*joints_position[1+bone].z + 1.0/2*joints_position[2+bone].z;

    //    finger_bone[0].at<float>(0,4) = joints_position[2+bone].x;
    //    finger_bone[0].at<float>(1,4) = 1.0/3*joints_position[1+bone].y + 2.0/3*joints_position[2+bone].y;
    //    finger_bone[0].at<float>(2,4) = 1.0/2*joints_position[1+bone].z + 1.0/2*joints_position[2+bone].z;

    //    finger_bone[0].at<float>(0,5) = joints_position[2+bone].x;
    //    finger_bone[0].at<float>(1,5) = 2.0/3*joints_position[1+bone].y + 1.0/3*joints_position[2+bone].y;
    //    finger_bone[0].at<float>(2,5) = 1.0/2*joints_position[1+bone].z + 1.0/2*joints_position[2+bone].z;

    //    //other fingers
    //    for(int i = 1; i< 5; i++){
    //        finger_bone[i] = Mat::zeros(3, 6, CV_32FC1);

    //        finger_bone[i].at<float>(0,0) = joints_position[i*5+bone+2].x;
    //        finger_bone[i].at<float>(1,0) = joints_position[i*5+bone+2].y;
    //        finger_bone[i].at<float>(2,0) = joints_position[i*5+bone+2].z;

    //        finger_bone[i].at<float>(0,1) = joints_position[i*5+bone+3].x;
    //        finger_bone[i].at<float>(1,1) = joints_position[i*5+bone+3].y;
    //        finger_bone[i].at<float>(2,1) = joints_position[i*5+bone+3].z;

    //        //        finger_bone[i].at<float>(0,1) = (joints_position[i*5+bone+3].x - joints_position[i*5+bone+2].x)/Distance_2XYZRGB(joints_position[i*5+bone+3], joints_position[i*5+bone+2])*bone_length[i][bone+1] + joints_position[i*5+bone+2].x;
    //        //        finger_bone[i].at<float>(1,1) = (joints_position[i*5+bone+3].y - joints_position[i*5+bone+2].y)/Distance_2XYZRGB(joints_position[i*5+bone+3], joints_position[i*5+bone+2])*bone_length[i][bone+1] + joints_position[i*5+bone+2].y;
    //        //        finger_bone[i].at<float>(2,1) = (joints_position[i*5+bone+3].z - joints_position[i*5+bone+2].z)/Distance_2XYZRGB(joints_position[i*5+bone+3], joints_position[i*5+bone+2])*bone_length[i][bone+1] + joints_position[i*5+bone+2].z;

    //        finger_bone[i].at<float>(0,2) = 1.0/3*joints_position[i*5+bone+2].x + 2.0/3*joints_position[i*5+bone+3].x;
    //        finger_bone[i].at<float>(1,2) = joints_position[i*5+bone+2].y;
    //        finger_bone[i].at<float>(2,2) = 1.0/2*joints_position[i*5+bone+2].z + 1.0/2*joints_position[i*5+bone+3].z;

    //        finger_bone[i].at<float>(0,3) = 2.0/3*joints_position[i*5+bone+2].x + 1.0/3*joints_position[i*5+bone+3].x;
    //        finger_bone[i].at<float>(1,3) = joints_position[i*5+bone+2].y;
    //        finger_bone[i].at<float>(2,3) = 1.0/2*joints_position[i*5+bone+2].z + 1.0/2*joints_position[i*5+bone+3].z;

    //        finger_bone[i].at<float>(0,4) = joints_position[i*5+bone+3].x;
    //        finger_bone[i].at<float>(1,4) = 1.0/3*joints_position[i*5+bone+2].y + 2.0/3*joints_position[i*5+bone+3].y;
    //        finger_bone[i].at<float>(2,4) = 1.0/2*joints_position[i*5+bone+2].z + 1.0/2*joints_position[i*5+bone+3].z;

    //        finger_bone[i].at<float>(0,5) = joints_position[i*5+bone+3].x;
    //        finger_bone[i].at<float>(1,5) = 2.0/3*joints_position[i*5+bone+2].y + 1.0/3*joints_position[i*5+bone+3].y;
    //        finger_bone[i].at<float>(2,5) = 1.0/2*joints_position[i*5+bone+2].z + 1.0/2*joints_position[i*5+bone+3].z;
    //    }
    Mat finger_bone_discrete[5];
    for(int i = 0; i<5; i++){
        finger_bone_discrete[i] = finger_bone[i]*1000/resolution+imageSize/2;
    }
    ROS_INFO("Here1");

    //1.2 find nearest neighbour:
    double temp_dis[5][6];
    for(int i = 0; i< 5; i++)
        for(int j = 0; j < 6; j++)
            temp_dis[i][j] = 10000;

    int temp_row[5][6], temp_col[5][6];
    std::cout << temp_row[0][1] <<std::endl;
    std::cout << "bone: " << bone <<std::endl;
    std::cout<< "finger_bone discrete: " << finger_bone_discrete[0] << std::endl;

    for( int row = 0; row < LabelMat.rows; row++){
        for(int col = 0; col < LabelMat.cols; col++){
            if(int(LabelMat.at<unsigned char>(row, col)) == 7)
                std::cout<< "row:" << row<< " " << col << " " << int(LabelMat.at<unsigned char>(row, col))-bone << std::endl;
            switch(int(LabelMat.at<unsigned char>(row, col))-bone){
            //thumb proximal
            case 5:
            {
                double dis = (Hand_DepthMat.at<unsigned char>(row, col) - finger_bone_discrete[0].at<float>(2,1)) * (Hand_DepthMat.at<unsigned char>(row, col) - finger_bone_discrete[0].at<float>(2,1))+
                        (row - finger_bone_discrete[0].at<float>(1,1)) * (row - finger_bone_discrete[0].at<float>(1,1)) +
                        (col - finger_bone_discrete[0].at<float>(0,1)) * (col - finger_bone_discrete[0].at<float>(0,1));
                std::cout<< "dis1: " <<dis <<" " << row<< " " << col << " " << int(Hand_DepthMat.at<unsigned char>(row, col)) << std::endl;
                if (dis < temp_dis[0][1]){
                    temp_row[0][1] = row;
                    temp_col[0][1]= col;
                    temp_dis[0][1] = dis;
                }

                dis = (Hand_DepthMat.at<unsigned char>(row, col) - finger_bone_discrete[0].at<float>(2,2)) * (Hand_DepthMat.at<unsigned char>(row, col) - finger_bone_discrete[0].at<float>(2,2))+
                        (row - finger_bone_discrete[0].at<float>(1,2)) * (row - finger_bone_discrete[0].at<float>(1,2)) +
                        (col - finger_bone_discrete[0].at<float>(0,2)) * (col - finger_bone_discrete[0].at<float>(0,2));
                if (dis < temp_dis[0][2]){
                    temp_row[0][2] = row;
                    temp_col[0][2]= col;
                    temp_dis[0][2] = dis;
                }

                dis = (Hand_DepthMat.at<unsigned char>(row, col) - finger_bone_discrete[0].at<float>(2,3)) * (Hand_DepthMat.at<unsigned char>(row, col) - finger_bone_discrete[0].at<float>(2,3))+
                        (row - finger_bone_discrete[0].at<float>(1,3)) * (row - finger_bone_discrete[0].at<float>(1,3)) +
                        (col - finger_bone_discrete[0].at<float>(0,3)) * (col - finger_bone_discrete[0].at<float>(0,3));
                if (dis < temp_dis[0][3]){
                    temp_row[0][3] = row;
                    temp_col[0][3]= col;
                    temp_dis[0][3] = dis;
                }

                dis = (Hand_DepthMat.at<unsigned char>(row, col) - finger_bone_discrete[0].at<float>(2,4)) * (Hand_DepthMat.at<unsigned char>(row, col) - finger_bone_discrete[0].at<float>(2,4))+
                        (row - finger_bone_discrete[0].at<float>(1,4)) * (row - finger_bone_discrete[0].at<float>(1,4)) +
                        (col - finger_bone_discrete[0].at<float>(0,4)) * (col - finger_bone_discrete[0].at<float>(0,4));
                if (dis < temp_dis[0][4]){
                    temp_row[0][4] = row;
                    temp_col[0][4]= col;
                    temp_dis[0][4] = dis;
                }

                dis = (Hand_DepthMat.at<unsigned char>(row, col) - finger_bone_discrete[0].at<float>(2,5)) * (Hand_DepthMat.at<unsigned char>(row, col) - finger_bone_discrete[0].at<float>(2,5))+
                        (row - finger_bone_discrete[0].at<float>(1,5)) * (row - finger_bone_discrete[0].at<float>(1,5)) +
                        (col - finger_bone_discrete[0].at<float>(0,5)) * (col - finger_bone_discrete[0].at<float>(0,5));
                if (dis < temp_dis[0][5]){
                    temp_row[0][5] = row;
                    temp_col[0][5]= col;
                    temp_dis[0][5] = dis;
                }

                break;
            }

                //index finger proximal
            case 8:
            {
                double dis = (Hand_DepthMat.at<unsigned char>(row, col) - finger_bone_discrete[1].at<float>(2,1)) * (Hand_DepthMat.at<unsigned char>(row, col) - finger_bone_discrete[1].at<float>(2,1))+
                        (row - finger_bone_discrete[1].at<float>(1,1)) * (row - finger_bone_discrete[1].at<float>(1,1)) +
                        (col - finger_bone_discrete[1].at<float>(0,1)) * (col - finger_bone_discrete[1].at<float>(0,1));
                if (dis < temp_dis[1][1]){
                    temp_row[1][1] = row;
                    temp_col[1][1]= col;
                    temp_dis[1][1] = dis;
                }

                dis = (Hand_DepthMat.at<unsigned char>(row, col) - finger_bone_discrete[1].at<float>(2,2)) * (Hand_DepthMat.at<unsigned char>(row, col) - finger_bone_discrete[1].at<float>(2,2))+
                        (row - finger_bone_discrete[1].at<float>(1,2)) * (row - finger_bone_discrete[1].at<float>(1,2)) +
                        (col - finger_bone_discrete[1].at<float>(0,2)) * (col - finger_bone_discrete[1].at<float>(0,2));
                if (dis < temp_dis[1][2]){
                    temp_row[1][2] = row;
                    temp_col[1][2]= col;
                    temp_dis[1][2] = dis;
                }

                dis = (Hand_DepthMat.at<unsigned char>(row, col) - finger_bone_discrete[1].at<float>(2,3)) * (Hand_DepthMat.at<unsigned char>(row, col) - finger_bone_discrete[1].at<float>(2,3))+
                        (row - finger_bone_discrete[1].at<float>(1,3)) * (row - finger_bone_discrete[1].at<float>(1,3)) +
                        (col - finger_bone_discrete[1].at<float>(0,3)) * (col - finger_bone_discrete[1].at<float>(0,3));
                if (dis < temp_dis[1][3]){
                    temp_row[1][3] = row;
                    temp_col[1][3]= col;
                    temp_dis[1][3] = dis;
                }

                dis = (Hand_DepthMat.at<unsigned char>(row, col) - finger_bone_discrete[1].at<float>(2,4)) * (Hand_DepthMat.at<unsigned char>(row, col) - finger_bone_discrete[1].at<float>(2,4))+
                        (row - finger_bone_discrete[1].at<float>(1,4)) * (row - finger_bone_discrete[1].at<float>(1,4)) +
                        (col - finger_bone_discrete[1].at<float>(0,4)) * (col - finger_bone_discrete[1].at<float>(0,4));
                if (dis < temp_dis[1][4]){
                    temp_row[1][4] = row;
                    temp_col[1][4]= col;
                    temp_dis[1][4] = dis;
                }

                dis = (Hand_DepthMat.at<unsigned char>(row, col) - finger_bone_discrete[1].at<float>(2,5)) * (Hand_DepthMat.at<unsigned char>(row, col) - finger_bone_discrete[1].at<float>(2,5))+
                        (row - finger_bone_discrete[1].at<float>(1,5)) * (row - finger_bone_discrete[1].at<float>(1,5)) +
                        (col - finger_bone_discrete[1].at<float>(0,5)) * (col - finger_bone_discrete[1].at<float>(0,5));
                if (dis < temp_dis[1][5]){
                    temp_row[1][5] = row;
                    temp_col[1][5]= col;
                    temp_dis[1][5] = dis;
                }

                break;
            }

                //middle finger proximal
            case 11:
            {
                double dis = (Hand_DepthMat.at<unsigned char>(row, col) - finger_bone_discrete[2].at<float>(2,1)) * (Hand_DepthMat.at<unsigned char>(row, col) - finger_bone_discrete[2].at<float>(2,1))+
                        (row - finger_bone_discrete[2].at<float>(1,1)) * (row - finger_bone_discrete[2].at<float>(1,1)) +
                        (col - finger_bone_discrete[2].at<float>(0,1)) * (col - finger_bone_discrete[2].at<float>(0,1));
                if (dis < temp_dis[2][1]){
                    temp_row[2][1] = row;
                    temp_col[2][1]= col;
                    temp_dis[2][1] = dis;
                }

                dis = (Hand_DepthMat.at<unsigned char>(row, col) - finger_bone_discrete[2].at<float>(2,2)) * (Hand_DepthMat.at<unsigned char>(row, col) - finger_bone_discrete[2].at<float>(2,2))+
                        (row - finger_bone_discrete[2].at<float>(1,2)) * (row - finger_bone_discrete[2].at<float>(1,2)) +
                        (col - finger_bone_discrete[2].at<float>(0,2)) * (col - finger_bone_discrete[2].at<float>(0,2));
                if (dis < temp_dis[2][2]){
                    temp_row[2][2] = row;
                    temp_col[2][2]= col;
                    temp_dis[2][2] = dis;
                }

                dis = (Hand_DepthMat.at<unsigned char>(row, col) - finger_bone_discrete[2].at<float>(2,3)) * (Hand_DepthMat.at<unsigned char>(row, col) - finger_bone_discrete[2].at<float>(2,3))+
                        (row - finger_bone_discrete[2].at<float>(1,3)) * (row - finger_bone_discrete[2].at<float>(1,3)) +
                        (col - finger_bone_discrete[2].at<float>(0,3)) * (col - finger_bone_discrete[2].at<float>(0,3));
                if (dis < temp_dis[2][3]){
                    temp_row[2][3] = row;
                    temp_col[2][3]= col;
                    temp_dis[2][3] = dis;
                }

                dis = (Hand_DepthMat.at<unsigned char>(row, col) - finger_bone_discrete[2].at<float>(2,4)) * (Hand_DepthMat.at<unsigned char>(row, col) - finger_bone_discrete[2].at<float>(2,4))+
                        (row - finger_bone_discrete[2].at<float>(1,4)) * (row - finger_bone_discrete[2].at<float>(1,4)) +
                        (col - finger_bone_discrete[2].at<float>(0,4)) * (col - finger_bone_discrete[2].at<float>(0,4));
                if (dis < temp_dis[2][4]){
                    temp_row[2][4] = row;
                    temp_col[2][4]= col;
                    temp_dis[2][4] = dis;
                }

                dis = (Hand_DepthMat.at<unsigned char>(row, col) - finger_bone_discrete[2].at<float>(2,5)) * (Hand_DepthMat.at<unsigned char>(row, col) - finger_bone_discrete[2].at<float>(2,5))+
                        (row - finger_bone_discrete[2].at<float>(1,5)) * (row - finger_bone_discrete[2].at<float>(1,5)) +
                        (col - finger_bone_discrete[2].at<float>(0,5)) * (col - finger_bone_discrete[2].at<float>(0,5));
                if (dis < temp_dis[2][5]){
                    temp_row[2][5] = row;
                    temp_col[2][5]= col;
                    temp_dis[2][5] = dis;
                }

                break;
            }

                //ring finger proximal
            case 14:
            {
                double dis = (Hand_DepthMat.at<unsigned char>(row, col) - finger_bone_discrete[3].at<float>(2,1)) * (Hand_DepthMat.at<unsigned char>(row, col) - finger_bone_discrete[3].at<float>(2,1))+
                        (row - finger_bone_discrete[3].at<float>(1,1)) * (row - finger_bone_discrete[3].at<float>(1,1)) +
                        (col - finger_bone_discrete[3].at<float>(0,1)) * (col - finger_bone_discrete[3].at<float>(0,1));
                //std::cout<< "dis1: " <<dis <<" " << row<< " " << col << " " << int(Hand_DepthMat.at<unsigned char>(row, col)) << std::endl;
                if (dis < temp_dis[3][1]){
                    temp_row[3][1] = row;
                    temp_col[3][1]= col;
                    temp_dis[3][1] = dis;
                }

                dis = (Hand_DepthMat.at<unsigned char>(row, col) - finger_bone_discrete[3].at<float>(2,2)) * (Hand_DepthMat.at<unsigned char>(row, col) - finger_bone_discrete[3].at<float>(2,2))+
                        (row - finger_bone_discrete[3].at<float>(1,2)) * (row - finger_bone_discrete[3].at<float>(1,2)) +
                        (col - finger_bone_discrete[3].at<float>(0,2)) * (col - finger_bone_discrete[3].at<float>(0,2));
                //std::cout<< "dis2: " <<dis <<std::endl;
                if (dis < temp_dis[3][2]){
                    temp_row[3][2] = row;
                    temp_col[3][2]= col;
                    temp_dis[3][2] = dis;
                }

                dis = (Hand_DepthMat.at<unsigned char>(row, col) - finger_bone_discrete[3].at<float>(2,3)) * (Hand_DepthMat.at<unsigned char>(row, col) - finger_bone_discrete[3].at<float>(2,3))+
                        (row - finger_bone_discrete[3].at<float>(1,3)) * (row - finger_bone_discrete[3].at<float>(1,3)) +
                        (col - finger_bone_discrete[3].at<float>(0,3)) * (col - finger_bone_discrete[3].at<float>(0,3));
                if (dis < temp_dis[3][3]){
                    temp_row[3][3] = row;
                    temp_col[3][3]= col;
                    temp_dis[3][3] = dis;
                }

                dis = (Hand_DepthMat.at<unsigned char>(row, col) - finger_bone_discrete[3].at<float>(2,4)) * (Hand_DepthMat.at<unsigned char>(row, col) - finger_bone_discrete[3].at<float>(2,4))+
                        (row - finger_bone_discrete[3].at<float>(1,4)) * (row - finger_bone_discrete[3].at<float>(1,4)) +
                        (col - finger_bone_discrete[3].at<float>(0,4)) * (col - finger_bone_discrete[3].at<float>(0,4));
                if (dis < temp_dis[3][4]){
                    temp_row[3][4] = row;
                    temp_col[3][4]= col;
                    temp_dis[3][4] = dis;
                }

                dis = (Hand_DepthMat.at<unsigned char>(row, col) - finger_bone_discrete[3].at<float>(2,5)) * (Hand_DepthMat.at<unsigned char>(row, col) - finger_bone_discrete[3].at<float>(2,5))+
                        (row - finger_bone_discrete[3].at<float>(1,5)) * (row - finger_bone_discrete[3].at<float>(1,5)) +
                        (col - finger_bone_discrete[3].at<float>(0,5)) * (col - finger_bone_discrete[3].at<float>(0,5));
                if (dis < temp_dis[3][5]){
                    temp_row[3][5] = row;
                    temp_col[3][5]= col;
                    temp_dis[3][5] = dis;
                }

                break;
            }

                //little finger proximal
            case 17:
            {
                double dis = (Hand_DepthMat.at<unsigned char>(row, col) - finger_bone_discrete[4].at<float>(2,1)) * (Hand_DepthMat.at<unsigned char>(row, col) - finger_bone_discrete[4].at<float>(2,1))+
                        (row - finger_bone_discrete[4].at<float>(1,1)) * (row - finger_bone_discrete[4].at<float>(1,1)) +
                        (col - finger_bone_discrete[4].at<float>(0,1)) * (col - finger_bone_discrete[4].at<float>(0,1));
                if (dis < temp_dis[4][1]){
                    temp_row[4][1] = row;
                    temp_col[4][1]= col;
                    temp_dis[4][1] = dis;
                }

                dis = (Hand_DepthMat.at<unsigned char>(row, col) - finger_bone_discrete[4].at<float>(2,2)) * (Hand_DepthMat.at<unsigned char>(row, col) - finger_bone_discrete[4].at<float>(2,2))+
                        (row - finger_bone_discrete[4].at<float>(1,2)) * (row - finger_bone_discrete[4].at<float>(1,2)) +
                        (col - finger_bone_discrete[4].at<float>(0,2)) * (col - finger_bone_discrete[4].at<float>(0,2));
                if (dis < temp_dis[4][2]){
                    temp_row[4][2] = row;
                    temp_col[4][2]= col;
                    temp_dis[4][2] = dis;
                }

                dis = (Hand_DepthMat.at<unsigned char>(row, col) - finger_bone_discrete[4].at<float>(2,3)) * (Hand_DepthMat.at<unsigned char>(row, col) - finger_bone_discrete[4].at<float>(2,3))+
                        (row - finger_bone_discrete[4].at<float>(1,3)) * (row - finger_bone_discrete[4].at<float>(1,3)) +
                        (col - finger_bone_discrete[4].at<float>(0,3)) * (col - finger_bone_discrete[4].at<float>(0,3));
                if (dis < temp_dis[4][3]){
                    temp_row[4][3] = row;
                    temp_col[4][3]= col;
                    temp_dis[4][3] = dis;
                }

                dis = (Hand_DepthMat.at<unsigned char>(row, col) - finger_bone_discrete[4].at<float>(2,4)) * (Hand_DepthMat.at<unsigned char>(row, col) - finger_bone_discrete[4].at<float>(2,4))+
                        (row - finger_bone_discrete[4].at<float>(1,4)) * (row - finger_bone_discrete[4].at<float>(1,4)) +
                        (col - finger_bone_discrete[4].at<float>(0,4)) * (col - finger_bone_discrete[4].at<float>(0,4));
                if (dis < temp_dis[4][4]){
                    temp_row[4][4] = row;
                    temp_col[4][4]= col;
                    temp_dis[4][4] = dis;
                }

                dis = (Hand_DepthMat.at<unsigned char>(row, col) - finger_bone_discrete[4].at<float>(2,5)) * (Hand_DepthMat.at<unsigned char>(row, col) - finger_bone_discrete[4].at<float>(2,5))+
                        (row - finger_bone_discrete[4].at<float>(1,5)) * (row - finger_bone_discrete[4].at<float>(1,5)) +
                        (col - finger_bone_discrete[4].at<float>(0,5)) * (col - finger_bone_discrete[4].at<float>(0,5));
                if (dis < temp_dis[4][5]){
                    temp_row[4][5] = row;
                    temp_col[4][5]= col;
                    temp_dis[4][5] = dis;
                }

            }

            default:
            {
                ;
            }

            }
        }
    }

    //1.3 estimate palm pose:
    //1.3.1 put nearest neighbour back to 3D space:
    //    ROS_INFO("Here!");
    //    for(int i = 0; i<5; i++){
    //        std::cout << "finger discrete " <<i<< ": " << finger_bone_discrete[i] <<std::endl;
    //    }
    ROS_INFO("Here2!");
    Mat Oberservation[5];
    for(int f = 0; f< 5; f++){
        Oberservation[f]= Mat::zeros(3, 6, CV_32FC1);

        for(int i = 1; i < 6; i++){
            Oberservation[f].at<float>(0,i) = (temp_col[f][i]-imageSize/2.0)*resolution/1000.0;
            Oberservation[f].at<float>(1,i) = (temp_row[f][i]-imageSize/2.0)*resolution/1000.0;
            std::cout<< f << ", " << i << ": (" << temp_row[f][i] << ", " << temp_col[f][i] << ") " << int(Hand_DepthMat.at<unsigned char>(temp_row[f][i], temp_col[f][i]))<<std::endl;
            Oberservation[f].at<float>(2,i) = (Hand_DepthMat.at<unsigned char>(temp_row[f][i], temp_col[f][i])-imageSize/2.0)*resolution/1000.0;
        }

    }
    ROS_INFO("Here3!");
    Oberservation[0].at<float>(0,0) = joints_position[1+bone].x;
    Oberservation[0].at<float>(1,0) = joints_position[1+bone].y;
    Oberservation[0].at<float>(2,0) = joints_position[1+bone].z;
    for(int f = 1; f< 5; f++){
        Oberservation[f].at<float>(0,0) = joints_position[f*5+bone+2].x;
        Oberservation[f].at<float>(1,0) = joints_position[f*5+bone+2].y;
        Oberservation[f].at<float>(2,0) = joints_position[f*5+bone+2].z;
    }


    //1.3.2 compute
    Mat R,t, p2, p1;
    p1 = Mat::zeros(3,1,CV_32FC1);
    p2 = Mat::zeros(3,1,CV_32FC1);
    poseEstimate::poseestimate::compute(Oberservation[0],finger_bone[0],R,t);
    p2.at<float>(0,0) = joints_position[2+bone].x;
    p2.at<float>(1,0) = joints_position[2+bone].y;
    p2.at<float>(2,0) = joints_position[2+bone].z;

    p1.at<float>(0,0) = joints_position[1+bone].x;
    p1.at<float>(1,0) = joints_position[1+bone].y;
    p1.at<float>(2,0) = joints_position[1+bone].z;
    Mat len = ((p2-p1).t() *(p2 - p1));
    p2 = R*(p2-p1)/sqrt(len.at<float>(0,0))*bone_length[0][bone]/1000.0+p1;

    joints_position[2+bone].x = p2.at<float>(0,0);
    joints_position[2+bone].y = p2.at<float>(1,0);
    joints_position[2+bone].z = p2.at<float>(2,0);

    for(int f = 1; f < 5; f++){

        poseEstimate::poseestimate::compute(Oberservation[f],finger_bone[f],R,t);
        p2.at<float>(0,0) = joints_position[f*5+bone+3].x;
        p2.at<float>(1,0) = joints_position[f*5+bone+3].y;
        p2.at<float>(2,0) = joints_position[f*5+bone+3].z;

        p1.at<float>(0,0) = joints_position[f*5+bone+2].x;
        p1.at<float>(1,0) = joints_position[f*5+bone+2].y;
        p1.at<float>(2,0) = joints_position[f*5+bone+2].z;

        len = ((p2-p1).t() *(p2 - p1));
        p2 = R*(p2-p1)/sqrt(len.at<float>(0,0))*bone_length[f][bone+1]/1000.0+p1;
        //std::cout << f << ", " << bone << ": " << (p2-p1).t() *(p2 - p1)  << std::endl;

        joints_position[f*5+bone+3].x = p2.at<float>(0,0);
        joints_position[f*5+bone+3].y = p2.at<float>(1,0);
        joints_position[f*5+bone+3].z = p2.at<float>(2,0);

    }

    if(bone == 2)
        joints_position[5] = joints_position[4];

}

void articulate_HandModel_XYZRGB::finger_fitting3(vector< vector<pcl::PointXYZ> > labelPointXYZ, int bone){
    /* initialize random seed: */
    srand (time(NULL));
    ////////////////////////Chapter 1 thumb:
    float ratio = 0, ratio_temp = 0;
    int count = 0, index_temp = 0;

    while(ratio < 0.8 && 2*count < labelPointXYZ[5 + bone].size()){
        //1. randomly select one point in label 5
        int point_index = rand() % labelPointXYZ[5 + bone].size();
        //2. calculate the line pass through joint position 1 and selected point

        Mat v = Mat::zeros(3,1,CV_32FC1);
        Mat w = Mat::zeros(3,1,CV_32FC1);
        v.at<float>(0,0) = labelPointXYZ[5 + bone][point_index].x - joints_position[1 + bone].x;
        v.at<float>(1,0) = labelPointXYZ[5 + bone][point_index].y - joints_position[1 + bone].y;
        v.at<float>(2,0) = labelPointXYZ[5 + bone][point_index].z - joints_position[1 + bone].z;

        int inliers = 0, outliers = 0;

        //3. calculate distance of all points in label 5 to the line and get inlier ratio
        for(int i = 0; i < labelPointXYZ[5 + bone].size(); i++){
            w.at<float>(0,0) = labelPointXYZ[5 + bone][i].x - joints_position[1 + bone].x;
            w.at<float>(1,0) = labelPointXYZ[5 + bone][i].y - joints_position[1 + bone].y;
            w.at<float>(2,0) = labelPointXYZ[5 + bone][i].z - joints_position[1 + bone].z;

            Mat c1 = w.t()*v;
            Mat c2 = v.t()*v;
            float distance;
            if( c1.at<float>(0,0) <= 0){
                distance = Distance_XYZXYZRGB(labelPointXYZ[5 + bone][i], joints_position[1 + bone]) - 0.001;
            }
            else if (c2.at<float>(0,0) <= c1.at<float>(0,0)){
                distance = Distance_2XYZ(labelPointXYZ[5 + bone][i],labelPointXYZ[5 + bone][point_index]);
            }
            else{
                float b = c1.at<float>(0,0)/c2.at<float>(0,0);
                pcl::PointXYZ p_xyz;
                p_xyz.x = joints_position[1 + bone].x + b*v.at<float>(0,0);
                p_xyz.y = joints_position[1 + bone].y + b*v.at<float>(1,0);
                p_xyz.z = joints_position[1 + bone].z + b*v.at<float>(2,0);
                distance = Distance_2XYZ(labelPointXYZ[5 + bone][i], p_xyz);

            }

            if(distance < 0.005 - bone * 0.0003)
                inliers++;
            else
                outliers++;

        }
        ratio = inliers*1.0/(outliers+ inliers);
        //        std::cout<< "Inlier: " << inliers << "  Outlier: " << outliers << std::endl;
        //        std::cout<< "Count: " << count << "Inlier ratio: " << ratio << std::endl;
        count++;

        if(ratio > ratio_temp){
            ratio_temp = ratio;
            index_temp = point_index;
        }
    }

    //    std::cout<< "Final: " << ratio_temp << "  Index: " << index_temp << std::endl;
    float fac = bone_length[0][0 + bone]/1000.0/Distance_XYZXYZRGB(labelPointXYZ[5 + bone][index_temp], joints_position[1 + bone]);
    joints_position[2 + bone].x = fac*(labelPointXYZ[5 + bone][index_temp].x - joints_position[1 + bone].x) + joints_position[1 + bone].x;
    joints_position[2 + bone].y = fac*(labelPointXYZ[5 + bone][index_temp].y - joints_position[1 + bone].y) + joints_position[1 + bone].y;
    joints_position[2 + bone].z = fac*(labelPointXYZ[5 + bone][index_temp].z - joints_position[1 + bone].z) + joints_position[1 + bone].z;

    joints_position[5] = joints_position[4];


    ////////////////////////Chapter 2 other fingers:
    //#pragma omp for
    for(int finger = 1; finger < 5; finger++){
        ratio = 0;
        ratio_temp = 0;
        count = 0;
        index_temp = 0;
        float sum_of_distance_temp = 10000;

        std::cout<< "   SIZE:  " << labelPointXYZ[5+3*finger + bone].size() << std::endl;

        while(ratio < 0.8 && 2*count < labelPointXYZ[5+3*finger + bone].size()){

            //1. randomly select one point in label 5
            int point_index = rand() % labelPointXYZ[5+3*finger + bone].size();
            //2. calculate the line pass through joint position 1 and selected point

            Mat v = Mat::zeros(3,1,CV_32FC1);
            Mat w = Mat::zeros(3,1,CV_32FC1);
            v.at<float>(0,0) = labelPointXYZ[5+3*finger + bone][point_index].x - joints_position[2+5*finger + bone].x;
            v.at<float>(1,0) = labelPointXYZ[5+3*finger + bone][point_index].y - joints_position[2+5*finger + bone].y;
            v.at<float>(2,0) = labelPointXYZ[5+3*finger + bone][point_index].z - joints_position[2+5*finger + bone].z;

            int inliers = 0, outliers = 0;

            //3. calculate distance of all points in label 5 to the line and get inlier ratio
            float sum_of_distance = 0;
            for(int i = 0; i < labelPointXYZ[5+3*finger + bone].size(); i++){
                w.at<float>(0,0) = labelPointXYZ[5+3*finger + bone][i].x - joints_position[2+5*finger + bone].x;
                w.at<float>(1,0) = labelPointXYZ[5+3*finger + bone][i].y - joints_position[2+5*finger + bone].y;
                w.at<float>(2,0) = labelPointXYZ[5+3*finger + bone][i].z - joints_position[2+5*finger + bone].z;

                Mat c1 = w.t()*v;
                Mat c2 = v.t()*v;
                float distance;
                if( c1.at<float>(0,0) <= 0){
                    distance = Distance_XYZXYZRGB(labelPointXYZ[5+3*finger + bone][i], joints_position[2+5*finger + bone]);
                }
                else if (c2.at<float>(0,0) <= c1.at<float>(0,0)){
                    distance = Distance_2XYZ(labelPointXYZ[5+3*finger + bone][i],labelPointXYZ[5+3*finger + bone][point_index]) + 0.002;
                }
                else{
                    float b = c1.at<float>(0,0)/c2.at<float>(0,0);
                    pcl::PointXYZ p_xyz;
                    p_xyz.x = joints_position[2+5*finger + bone].x + b*v.at<float>(0,0);
                    p_xyz.y = joints_position[2+5*finger + bone].y + b*v.at<float>(1,0);
                    p_xyz.z = joints_position[2+5*finger + bone].z + b*v.at<float>(2,0);
                    distance = Distance_2XYZ(labelPointXYZ[5+3*finger + bone][i], p_xyz);

                }

                if(distance < 0.005 ){
                    sum_of_distance += distance*distance;
                    inliers++;
                }
                else
                    outliers++;

            }
            ratio = inliers*1.0/(outliers+ inliers);
            //            std::cout<< "Inlier: " << inliers << "  Outlier: " << outliers << std::endl;
            //            std::cout<< "Count: " << count << "Inlier ratio: " << ratio << std::endl;
            count++;

            if(ratio > ratio_temp /*|| sum_of_distance/inliers < sum_of_distance_temp*/){
                sum_of_distance_temp = sum_of_distance/inliers;
                ratio_temp = ratio;
                index_temp = point_index;
            }
        }

        //        std::cout<< "!!!!!!!!!!Final: " << ratio_temp << "  Index: " << index_temp << std::endl;
        float fac = bone_length[0 + finger][1 + bone]/1000.0/Distance_XYZXYZRGB(labelPointXYZ[5+3*finger + bone][index_temp], joints_position[2+5*finger + bone]);
        joints_position[3+5*finger + bone].x = labelPointXYZ[5+3*finger + bone][index_temp].x;
        joints_position[3+5*finger + bone].y = labelPointXYZ[5+3*finger + bone][index_temp].y;
        joints_position[3+5*finger + bone].z = labelPointXYZ[5+3*finger + bone][index_temp].z;

        //        joints_position[3+5*finger + bone].x = fac*(labelPointXYZ[5+3*finger + bone][index_temp].x - joints_position[2+5*finger + bone].x) + joints_position[2+5*finger + bone].x;
        //        joints_position[3+5*finger + bone].y = fac*(labelPointXYZ[5+3*finger + bone][index_temp].y - joints_position[2+5*finger + bone].y) + joints_position[2+5*finger + bone].y;
        //        joints_position[3+5*finger + bone].z = fac*(labelPointXYZ[5+3*finger + bone][index_temp].z - joints_position[2+5*finger + bone].z) + joints_position[2+5*finger + bone].z;

    }



}

void articulate_HandModel_XYZRGB::finger_fitting4(Mat Hand_DepthMat, Mat LabelMat, int resolution, int bone){
    vector< vector <Point3d> > Intersection(10, vector<Point3d>() );
    int imageSize = 300/resolution;
    //1. Looking for intersection region
    for(int row = 0; row < LabelMat.rows; row++){
        for(int col = 0; col < LabelMat.cols; col++){
            if( LabelMat.at<unsigned char>(row, col) != 0 && LabelMat.at<unsigned char>(row, col) != 3 && LabelMat.at<unsigned char>(row, col)%3 == 0){
                int L = LabelMat.at<unsigned char>(row, col);
                if(LabelMat.at<unsigned char>(row, col) - LabelMat.at<unsigned char>(row +1, col) == 1 ||  LabelMat.at<unsigned char>(row, col) - LabelMat.at<unsigned char>(row-1, col) == 1||
                        LabelMat.at<unsigned char>(row, col) - LabelMat.at<unsigned char>(row, col + 1) == 1 || LabelMat.at<unsigned char>(row, col) - LabelMat.at<unsigned char>(row, col - 1) == 1){
                    Point3d p3d;
                    p3d.x = (col-imageSize/2.0)*resolution/1000.0;
                    p3d.y = (row-imageSize/2.0)*resolution/1000.0;
                    p3d.z = (Hand_DepthMat.at<unsigned char>(row, col)-imageSize/2.0)*resolution/1000.0;
                    Intersection[(L/3-1)*2-2].push_back(p3d);
                }
                else if (LabelMat.at<unsigned char>(row, col) - LabelMat.at<unsigned char>(row +1, col) == -1 ||  LabelMat.at<unsigned char>(row, col) - LabelMat.at<unsigned char>(row-1, col) == -1||
                         LabelMat.at<unsigned char>(row, col) - LabelMat.at<unsigned char>(row, col + 1) == -1 || LabelMat.at<unsigned char>(row, col) - LabelMat.at<unsigned char>(row, col - 1) == -1){
                    Point3d p3d;
                    p3d.x = (col-imageSize/2.0)*resolution/1000.0;
                    p3d.y = (row-imageSize/2.0)*resolution/1000.0;
                    p3d.z = (Hand_DepthMat.at<unsigned char>(row, col)-imageSize/2.0)*resolution/1000.0;
                    Intersection[(L/3-1)*2-1].push_back(p3d);

                }

            }
        }

    }
    for(int i = 0; i < 10; i++)
        std::cout<< "Size of " << i << ": " << Intersection[i].size() << std::endl;
    //1.2 Ransac to find the intersection center:
    Point3d center[10];
    vector<int> invalid_index;
    for(int i = 0; i < 10; i++){
        if(Intersection[i].size()!=0)
            Ransac(Intersection[i], center[i], 10, 0.007);
        else
            invalid_index.push_back(i);
        //std::cout<< "Center " << i << ": " << center[i] << std::endl;
    }
    //2. Fitting 3 lines together

    //2.1 fitting first two joints to the mean center;
    joints_position[2].x = center[0].x;
    joints_position[2].y = center[0].y;
    joints_position[2].z = center[0].z;

    joints_position[3].x = center[1].x;
    joints_position[3].y = center[1].y;
    joints_position[3].z = center[1].z;

    for(int i = 1; i < 5; i++){

        joints_position[3+5*i].x = center[i*2].x;
        joints_position[3+5*i].y = center[i*2].y;
        joints_position[3+5*i].z = center[i*2].z;

        joints_position[4+5*i].x = center[i*2+1].x;
        joints_position[4+5*i].y = center[i*2+1].y;
        joints_position[4+5*i].z = center[i*2+1].z;
    }
    //2.2 fitting last joints(finger tip) according to the pcl;

}

void articulate_HandModel_XYZRGB::finger_fitting5(Mat Hand_DepthMat, Mat LabelMat, int resolution, int iteration_number, vector< vector<pcl::PointXYZ> > labelPointXYZ){
    vector< vector <Point3d> > Intersection(10, vector<Point3d>() );
    vector< vector <Point3d> > Distal_edge(5, vector<Point3d>() );
    int imageSize = 300/resolution;

    std::cout << "iteration_number: " << iteration_number << std::endl;
    srand((unsigned)time(NULL));

    //1. Looking for intersection region
    for(int row = 0; row < LabelMat.rows; row++){
        for(int col = 0; col < LabelMat.cols; col++){
            //intersection area:
            if( LabelMat.at<unsigned char>(row, col) != 0 && LabelMat.at<unsigned char>(row, col) != 3 && LabelMat.at<unsigned char>(row, col)%3 == 0){
                int L = LabelMat.at<unsigned char>(row, col);
                if(LabelMat.at<unsigned char>(row, col) - LabelMat.at<unsigned char>(row +1, col) == 1 ||  LabelMat.at<unsigned char>(row, col) - LabelMat.at<unsigned char>(row-1, col) == 1||
                        LabelMat.at<unsigned char>(row, col) - LabelMat.at<unsigned char>(row, col + 1) == 1 || LabelMat.at<unsigned char>(row, col) - LabelMat.at<unsigned char>(row, col - 1) == 1){
                    Point3d p3d;
                    p3d.x = (col-imageSize/2.0)*resolution/1000.0;
                    p3d.y = (row-imageSize/2.0)*resolution/1000.0;
                    p3d.z = (Hand_DepthMat.at<unsigned char>(row, col)-imageSize/2.0)*resolution/1000.0;
                    Intersection[(L/3-1)*2-2].push_back(p3d);
                }
                else if (LabelMat.at<unsigned char>(row, col) - LabelMat.at<unsigned char>(row +1, col) == -1 ||  LabelMat.at<unsigned char>(row, col) - LabelMat.at<unsigned char>(row-1, col) == -1||
                         LabelMat.at<unsigned char>(row, col) - LabelMat.at<unsigned char>(row, col + 1) == -1 || LabelMat.at<unsigned char>(row, col) - LabelMat.at<unsigned char>(row, col - 1) == -1){
                    Point3d p3d;
                    p3d.x = (col-imageSize/2.0)*resolution/1000.0;
                    p3d.y = (row-imageSize/2.0)*resolution/1000.0;
                    p3d.z = (Hand_DepthMat.at<unsigned char>(row, col)-imageSize/2.0)*resolution/1000.0;
                    Intersection[(L/3-1)*2-1].push_back(p3d);

                }

            }
            //distal edge:
            else if ( LabelMat.at<unsigned char>(row, col) != 1 && LabelMat.at<unsigned char>(row, col) != 4 && LabelMat.at<unsigned char>(row, col)%3 == 1){
                int L = LabelMat.at<unsigned char>(row, col);
                if(LabelMat.at<unsigned char>(row +1, col) == 0 || LabelMat.at<unsigned char>(row -1, col) == 0
                        || LabelMat.at<unsigned char>(row, col + 1) == 0 || LabelMat.at<unsigned char>(row, col - 1) == 0){
                    Point3d p3d;
                    p3d.x = (col-imageSize/2.0)*resolution/1000.0;
                    p3d.y = (row-imageSize/2.0)*resolution/1000.0;
                    p3d.z = (Hand_DepthMat.at<unsigned char>(row, col)-imageSize/2.0)*resolution/1000.0;
                    Distal_edge[(L-1)/3-2].push_back(p3d);
                }
            }
        }

    }
    //    for(int i = 0; i < 10; i++)
    //        std::cout<< "Size of " << i << ": " << Intersection[i].size() << std::endl;
    //1.2 Ransac to find the intersection center:
    Point3d center[10];
    vector<int> invalid_index;
    for(int i = 0; i < 10; i++){
        if(Intersection[i].size()!=0)
            Ransac(Intersection[i], center[i], 10, 0.007);
        else
            invalid_index.push_back(i);
        //std::cout<< "Center " << i << ": " << center[i] << std::endl;
    }

    //2. Fitting 3 lines together for thumb
    float T_ratio = 0.0, T_sum = 1000;
    for(int iteration = 0; iteration < iteration_number; iteration++){
        //2.1 randomly select two points in intersection areas:
        Point3d temp_joints_position[5][3];

        temp_joints_position[0][0].x = center[0].x+rand()%5/1000;
        temp_joints_position[0][0].y = center[0].y+rand()%5/1000;
        temp_joints_position[0][0].z = center[0].z+rand()%5/1000;

        temp_joints_position[0][1].x = center[1].x+rand()%5/1000;
        temp_joints_position[0][1].y = center[1].y+rand()%5/1000;
        temp_joints_position[0][1].z = center[1].z+rand()%5/1000;

        //2.2 randomly select one point in end of distal areas:
        float distance = 0.0;
        int count = 0;
        while(count < 10){
            int index = rand()% int( Distal_edge[0].size() );
            count ++;
            float temp_distance = Distance_2Point3d(Distal_edge[0][index], temp_joints_position[0][1]);
            if(temp_distance > distance){
                distance = temp_distance;
                temp_joints_position[0][2] = Distal_edge[0][index];
            }
        }

        //2.3calculate the distance as score:
        float sum = 0;
        int inlier_count = 0;
        for(int bone_index = 0; bone_index < 3; ++bone_index){
            //first thumb bone:
            if( bone_index  == 0){

                Mat v = Mat::zeros(3,1,CV_32FC1);
                Mat w = Mat::zeros(3,1,CV_32FC1);
                v.at<float>(0,0) = temp_joints_position[0][0].x - joints_position[1].x;
                v.at<float>(1,0) = temp_joints_position[0][0].y - joints_position[1].y;
                v.at<float>(2,0) = temp_joints_position[0][0].z - joints_position[1].z;

                //calculate distance of all points in label 5 to the line
                for(int i = 0; i < labelPointXYZ[5].size(); i++){
                    w.at<float>(0,0) = labelPointXYZ[5][i].x - joints_position[1].x;
                    w.at<float>(1,0) = labelPointXYZ[5][i].y - joints_position[1].y;
                    w.at<float>(2,0) = labelPointXYZ[5][i].z - joints_position[1].z;

                    Mat c1 = w.t()*v;
                    Mat c2 = v.t()*v;
                    float indivi_distance;
                    int outlier = 0;
                    if( c1.at<float>(0,0) <= 0){
                        //indivi_distance = Distance_XYZXYZRGB(labelPointXYZ[5][i], joints_position[1]);
                        outlier = 1;
                    }
                    else if (c2.at<float>(0,0) <= c1.at<float>(0,0)){
                        //indivi_distance = Distance_XYZPoint3d(labelPointXYZ[5][i],temp_joints_position[0][0]);
                        outlier = 1;
                    }
                    else{
                        float b = c1.at<float>(0,0)/c2.at<float>(0,0);
                        pcl::PointXYZ p_xyz;
                        p_xyz.x = joints_position[1].x + b*v.at<float>(0,0);
                        p_xyz.y = joints_position[1].y + b*v.at<float>(1,0);
                        p_xyz.z = joints_position[1].z + b*v.at<float>(2,0);
                        indivi_distance = Distance_2XYZ(labelPointXYZ[5][i], p_xyz);
                        sum += indivi_distance;
                    }


                    if(indivi_distance < 0.005)
                        inlier_count = inlier_count + 1 - outlier;
                }

            }
            //other thumb bone:
            else{
                Mat v = Mat::zeros(3,1,CV_32FC1);
                Mat w = Mat::zeros(3,1,CV_32FC1);
                v.at<float>(0,0) = temp_joints_position[0][bone_index].x - temp_joints_position[0][bone_index-1].x;
                v.at<float>(1,0) = temp_joints_position[0][bone_index].y - temp_joints_position[0][bone_index-1].y;
                v.at<float>(2,0) = temp_joints_position[0][bone_index].z - temp_joints_position[0][bone_index-1].z;

                //calculate distance of all points in label 5 to the line
                for(int i = 0; i < labelPointXYZ[5+bone_index].size(); i++){
                    w.at<float>(0,0) = labelPointXYZ[5+bone_index][i].x - temp_joints_position[0][bone_index-1].x;
                    w.at<float>(1,0) = labelPointXYZ[5+bone_index][i].y - temp_joints_position[0][bone_index-1].y;
                    w.at<float>(2,0) = labelPointXYZ[5+bone_index][i].z - temp_joints_position[0][bone_index-1].z;

                    Mat c1 = w.t()*v;
                    Mat c2 = v.t()*v;
                    float indivi_distance;
                    int outlier = 0;
                    if( c1.at<float>(0,0) <= 0){
                        //indivi_distance = Distance_XYZPoint3d(labelPointXYZ[5+bone_index][i], temp_joints_position[0][bone_index-1]);
                        outlier = 1;
                    }
                    else if (c2.at<float>(0,0) <= c1.at<float>(0,0)){
                        //indivi_distance = Distance_XYZPoint3d(labelPointXYZ[5+bone_index][i],temp_joints_position[0][bone_index]);
                        outlier = 1;
                    }
                    else{
                        float b = c1.at<float>(0,0)/c2.at<float>(0,0);
                        pcl::PointXYZ p_xyz;
                        p_xyz.x = temp_joints_position[0][bone_index-1].x + b*v.at<float>(0,0);
                        p_xyz.y = temp_joints_position[0][bone_index-1].y + b*v.at<float>(1,0);
                        p_xyz.z = temp_joints_position[0][bone_index-1].z + b*v.at<float>(2,0);
                        indivi_distance = Distance_2XYZ(labelPointXYZ[5+bone_index][i], p_xyz);
                        sum += indivi_distance;

                    }


                    if(indivi_distance < 0.005)
                        inlier_count = inlier_count + 1 - outlier;
                }
            }
        }
        float inlier_ratio = inlier_count*1.0/(labelPointXYZ[5].size() + labelPointXYZ[6].size() + labelPointXYZ[7].size());
        //std::cout << "inlier ratio: " << inlier_ratio << "    Average dis: " << sum/inlier_count << std::endl;
        if(inlier_ratio > T_ratio && sum/inlier_count < T_sum){
            joints_position[2].x = temp_joints_position[0][0].x;
            joints_position[2].y = temp_joints_position[0][0].y;
            joints_position[2].z = temp_joints_position[0][0].z;

            joints_position[3].x = temp_joints_position[0][1].x;
            joints_position[3].y = temp_joints_position[0][1].y;
            joints_position[3].z = temp_joints_position[0][1].z;

            joints_position[4].x = temp_joints_position[0][2].x;
            joints_position[4].y = temp_joints_position[0][2].y;
            joints_position[4].z = temp_joints_position[0][2].z;

            joints_position[5] = joints_position[4];
            T_ratio = inlier_ratio;
            T_sum = sum/inlier_count;
        }

    }


    //2. Fitting 3 lines together for other fingers:
    for(int finger_index = 1; finger_index < 5; ++ finger_index){
        T_ratio = 0.0;
        T_sum = 1000;
        for(int iteration = 0; iteration < iteration_number; iteration++){
            //2.1 randomly select two points in intersection areas:
            Point3d temp_joints_position[5][3];
            temp_joints_position[finger_index][0].x = center[2*finger_index].x+rand()%5/1000;
            temp_joints_position[finger_index][0].y = center[2*finger_index].y+rand()%5/1000;
            temp_joints_position[finger_index][0].z = center[2*finger_index].z+rand()%5/1000;

            temp_joints_position[finger_index][1].x = center[1+2*finger_index].x+rand()%5/1000;
            temp_joints_position[finger_index][1].y = center[1+2*finger_index].y+rand()%5/1000;
            temp_joints_position[finger_index][1].z = center[1+2*finger_index].z+rand()%5/1000;

            //2.2 randomly select one point in end of distal areas:
            float distance = 0.0;
            int count = 0;
            while(count < 7){
                int index = rand()% int( Distal_edge[finger_index].size() );
                count ++;
                float temp_distance = Distance_2Point3d(Distal_edge[finger_index][index], temp_joints_position[finger_index][1]);
                if(temp_distance > distance){
                    distance = temp_distance;
                    temp_joints_position[finger_index][2] = Distal_edge[finger_index][index];
                }
            }

            //2.3calculate the distance as score:
            float sum = 0;
            int inlier_count = 0;
            for(int bone_index = 0; bone_index < 3; ++bone_index){
                //first finger bone:
                if( bone_index  == 0){

                    Mat v = Mat::zeros(3,1,CV_32FC1);
                    Mat w = Mat::zeros(3,1,CV_32FC1);
                    v.at<float>(0,0) = temp_joints_position[finger_index][0].x - joints_position[5*finger_index+2].x;
                    v.at<float>(1,0) = temp_joints_position[finger_index][0].y - joints_position[5*finger_index+2].y;
                    v.at<float>(2,0) = temp_joints_position[finger_index][0].z - joints_position[5*finger_index+2].z;

                    //calculate distance of all points in label 5 to the line
                    for(int i = 0; i < labelPointXYZ[3*finger_index+5].size(); i++){
                        w.at<float>(0,0) = labelPointXYZ[3*finger_index+5][i].x - joints_position[5*finger_index+2].x;
                        w.at<float>(1,0) = labelPointXYZ[3*finger_index+5][i].y - joints_position[5*finger_index+2].y;
                        w.at<float>(2,0) = labelPointXYZ[3*finger_index+5][i].z - joints_position[5*finger_index+2].z;

                        Mat c1 = w.t()*v;
                        Mat c2 = v.t()*v;
                        float indivi_distance;
                        int outlier = 0;
                        if( c1.at<float>(0,0) <= 0){
                            //indivi_distance = Distance_XYZXYZRGB(labelPointXYZ[3*finger_index+5][i], joints_position[5*finger_index+2]);
                            outlier = 1;
                        }
                        else if (c2.at<float>(0,0) <= c1.at<float>(0,0)){
                            //indivi_distance = Distance_XYZPoint3d(labelPointXYZ[3*finger_index+5][i],temp_joints_position[finger_index][0]);
                            outlier = 1;
                        }
                        else{
                            float b = c1.at<float>(0,0)/c2.at<float>(0,0);
                            pcl::PointXYZ p_xyz;
                            p_xyz.x = joints_position[5*finger_index+2].x + b*v.at<float>(0,0);
                            p_xyz.y = joints_position[5*finger_index+2].y + b*v.at<float>(1,0);
                            p_xyz.z = joints_position[5*finger_index+2].z + b*v.at<float>(2,0);
                            indivi_distance = Distance_2XYZ(labelPointXYZ[3*finger_index+5][i], p_xyz);
                            sum += indivi_distance;
                        }

                        if(indivi_distance < 0.004)
                            inlier_count = inlier_count + 1 - outlier;
                    }

                }
                //other finger bone:
                else{
                    Mat v = Mat::zeros(3,1,CV_32FC1);
                    Mat w = Mat::zeros(3,1,CV_32FC1);
                    v.at<float>(0,0) = temp_joints_position[finger_index][bone_index].x - temp_joints_position[finger_index][bone_index-1].x;
                    v.at<float>(1,0) = temp_joints_position[finger_index][bone_index].y - temp_joints_position[finger_index][bone_index-1].y;
                    v.at<float>(2,0) = temp_joints_position[finger_index][bone_index].z - temp_joints_position[finger_index][bone_index-1].z;

                    //calculate distance of all points in label 5 to the line
                    for(int i = 0; i < labelPointXYZ[3*finger_index+5+bone_index].size(); i++){
                        w.at<float>(0,0) = labelPointXYZ[3*finger_index+5+bone_index][i].x - temp_joints_position[finger_index][bone_index-1].x;
                        w.at<float>(1,0) = labelPointXYZ[3*finger_index+5+bone_index][i].y - temp_joints_position[finger_index][bone_index-1].y;
                        w.at<float>(2,0) = labelPointXYZ[3*finger_index+5+bone_index][i].z - temp_joints_position[finger_index][bone_index-1].z;

                        Mat c1 = w.t()*v;
                        Mat c2 = v.t()*v;
                        float indivi_distance;
                        int outlier = 0;
                        if( c1.at<float>(0,0) <= 0){
                            //indivi_distance = Distance_XYZPoint3d(labelPointXYZ[3*finger_index+5+bone_index][i], temp_joints_position[finger_index][bone_index-1]);
                            outlier = 1;
                        }
                        else if (c2.at<float>(0,0) <= c1.at<float>(0,0)){
                            //indivi_distance = Distance_XYZPoint3d(labelPointXYZ[3*finger_index+5+bone_index][i],temp_joints_position[finger_index][bone_index]);
                            outlier = 1;
                        }
                        else{
                            float b = c1.at<float>(0,0)/c2.at<float>(0,0);
                            pcl::PointXYZ p_xyz;
                            p_xyz.x = temp_joints_position[finger_index][bone_index-1].x + b*v.at<float>(0,0);
                            p_xyz.y = temp_joints_position[finger_index][bone_index-1].y + b*v.at<float>(1,0);
                            p_xyz.z = temp_joints_position[finger_index][bone_index-1].z + b*v.at<float>(2,0);
                            indivi_distance = Distance_2XYZ(labelPointXYZ[3*finger_index+5+bone_index][i], p_xyz);
                            sum += indivi_distance;
                        }

                        if(indivi_distance < 0.003)
                            inlier_count = inlier_count + 1 - outlier;
                    }
                }
            }
            float inlier_ratio = inlier_count*1.0/(labelPointXYZ[3*finger_index+5].size() + labelPointXYZ[3*finger_index+6].size() + labelPointXYZ[3*finger_index+7].size());
            //std::cout << "inlier ratio: " << inlier_ratio << "    Average dis: " << sum/inlier_count << std::endl;
            if(inlier_ratio > T_ratio && sum/inlier_count < T_sum){
                joints_position[5*finger_index+3].x = temp_joints_position[finger_index][0].x;
                joints_position[5*finger_index+3].y = temp_joints_position[finger_index][0].y;
                joints_position[5*finger_index+3].z = temp_joints_position[finger_index][0].z;

                joints_position[5*finger_index+4].x = temp_joints_position[finger_index][1].x;
                joints_position[5*finger_index+4].y = temp_joints_position[finger_index][1].y;
                joints_position[5*finger_index+4].z = temp_joints_position[finger_index][1].z;

                joints_position[5*finger_index+5].x = temp_joints_position[finger_index][2].x;
                joints_position[5*finger_index+5].y = temp_joints_position[finger_index][2].y;
                joints_position[5*finger_index+5].z = temp_joints_position[finger_index][2].z;

                T_ratio = inlier_ratio;
                T_sum = sum/inlier_count;
            }

        }
    }

    //    //2.1 fitting first two joints to the mean center;
    //    joints_position[2].x = center[0].x;
    //    joints_position[2].y = center[0].y;
    //    joints_position[2].z = center[0].z;

    //    joints_position[3].x = center[1].x;
    //    joints_position[3].y = center[1].y;
    //    joints_position[3].z = center[1].z;

    //    for(int i = 1; i < 5; i++){

    //        joints_position[3+5*i].x = center[i*2].x;
    //        joints_position[3+5*i].y = center[i*2].y;
    //        joints_position[3+5*i].z = center[i*2].z;

    //        joints_position[4+5*i].x = center[i*2+1].x;
    //        joints_position[4+5*i].y = center[i*2+1].y;
    //        joints_position[4+5*i].z = center[i*2+1].z;
    //    }
    //    //2.2 fitting last joints(finger tip) according to the pcl;






}

void articulate_HandModel_XYZRGB::finger_fitting6(Mat const Hand_DepthMat, Mat const LabelMat, int const resolution, int const iteration_number, const vector< vector<pcl::PointXYZ> > labelPointXYZ, int const finger2fit){
    vector< vector <Point3d> > Intersection(10, vector<Point3d>() );
    vector< vector <Point3d> > Distal_edge(5, vector<Point3d>() );
    int imageSize = 300/resolution;

    std::cout << "iteration_number: " << iteration_number << std::endl;
    srand((unsigned)time(NULL));

    //1. Looking for intersection region
    for(int row = 0; row < LabelMat.rows; row++){
        for(int col = 0; col < LabelMat.cols; col++){
            //intersection area:
            if( LabelMat.at<unsigned char>(row, col) != 0 && LabelMat.at<unsigned char>(row, col) != 3 && LabelMat.at<unsigned char>(row, col)%3 == 0){
                int L = LabelMat.at<unsigned char>(row, col);
                if(LabelMat.at<unsigned char>(row, col) - LabelMat.at<unsigned char>(row +1, col) == 1 ||  LabelMat.at<unsigned char>(row, col) - LabelMat.at<unsigned char>(row-1, col) == 1||
                        LabelMat.at<unsigned char>(row, col) - LabelMat.at<unsigned char>(row, col + 1) == 1 || LabelMat.at<unsigned char>(row, col) - LabelMat.at<unsigned char>(row, col - 1) == 1){
                    Point3d p3d;
                    p3d.x = (col-imageSize/2.0)*resolution/1000.0;
                    p3d.y = (row-imageSize/2.0)*resolution/1000.0;
                    p3d.z = (Hand_DepthMat.at<unsigned char>(row, col)-imageSize/2.0)*resolution/1000.0;
                    Intersection[(L/3-1)*2-2].push_back(p3d);
                }
                else if (LabelMat.at<unsigned char>(row, col) - LabelMat.at<unsigned char>(row +1, col) == -1 ||  LabelMat.at<unsigned char>(row, col) - LabelMat.at<unsigned char>(row-1, col) == -1||
                         LabelMat.at<unsigned char>(row, col) - LabelMat.at<unsigned char>(row, col + 1) == -1 || LabelMat.at<unsigned char>(row, col) - LabelMat.at<unsigned char>(row, col - 1) == -1){
                    Point3d p3d;
                    p3d.x = (col-imageSize/2.0)*resolution/1000.0;
                    p3d.y = (row-imageSize/2.0)*resolution/1000.0;
                    p3d.z = (Hand_DepthMat.at<unsigned char>(row, col)-imageSize/2.0)*resolution/1000.0;
                    Intersection[(L/3-1)*2-1].push_back(p3d);

                }

            }
            //distal edge:
            else if ( LabelMat.at<unsigned char>(row, col) != 1 && LabelMat.at<unsigned char>(row, col) != 4 && LabelMat.at<unsigned char>(row, col)%3 == 1){
                int L = LabelMat.at<unsigned char>(row, col);
                if(LabelMat.at<unsigned char>(row +1, col) == 0 || LabelMat.at<unsigned char>(row -1, col) == 0
                        || LabelMat.at<unsigned char>(row, col + 1) == 0 || LabelMat.at<unsigned char>(row, col - 1) == 0){
                    Point3d p3d;
                    p3d.x = (col-imageSize/2.0)*resolution/1000.0;
                    p3d.y = (row-imageSize/2.0)*resolution/1000.0;
                    p3d.z = (Hand_DepthMat.at<unsigned char>(row, col)-imageSize/2.0)*resolution/1000.0;
                    Distal_edge[(L-1)/3-2].push_back(p3d);
                }
            }
        }

    }
    //    for(int i = 0; i < 10; i++)
    //        std::cout<< "Size of " << i << ": " << Intersection[i].size() << std::endl;
    //1.2 Ransac to find the intersection center:
    Point3d center[10];
    vector<int> invalid_index;
    for(int i = 0; i < 10; i++){
        if(Intersection[i].size()!=0)
            Ransac(Intersection[i], center[i], 10, 0.007);
        else
            invalid_index.push_back(i);
        //std::cout<< "Center " << i << ": " << center[i] << std::endl;
    }

    //2. Fitting 3 lines together for thumb
    float T_ratio = 0.0, T_sum = 1000;
    if(finger2fit == 1){
        for(int iteration = 0; iteration < iteration_number; iteration++){
            //2.1 randomly select two points in intersection areas:
            Point3d temp_joints_position[5][3];

            temp_joints_position[0][0].x = center[0].x+rand()%5/1000;
            temp_joints_position[0][0].y = center[0].y+rand()%5/1000;
            temp_joints_position[0][0].z = center[0].z+rand()%5/1000;

            temp_joints_position[0][1].x = center[1].x+rand()%5/1000;
            temp_joints_position[0][1].y = center[1].y+rand()%5/1000;
            temp_joints_position[0][1].z = center[1].z+rand()%5/1000;

            //2.2 randomly select one point in end of distal areas:
            float distance = 0.0;
            int count = 0;
            while(count < 10){
                int index = rand()% int( Distal_edge[0].size() );
                count ++;
                float temp_distance = Distance_2Point3d(Distal_edge[0][index], temp_joints_position[0][1]);
                if(temp_distance > distance){
                    distance = temp_distance;
                    temp_joints_position[0][2] = Distal_edge[0][index];
                }
            }

            //2.3calculate the distance as score:
            float sum = 0;
            int inlier_count = 0;
            for(int bone_index = 0; bone_index < 3; ++bone_index){
                //first thumb bone:
                if( bone_index  == 0){

                    Mat v = Mat::zeros(3,1,CV_32FC1);
                    Mat w = Mat::zeros(3,1,CV_32FC1);
                    v.at<float>(0,0) = temp_joints_position[0][0].x - joints_position[1].x;
                    v.at<float>(1,0) = temp_joints_position[0][0].y - joints_position[1].y;
                    v.at<float>(2,0) = temp_joints_position[0][0].z - joints_position[1].z;

                    //calculate distance of all points in label 5 to the line
                    for(int i = 0; i < labelPointXYZ[5].size(); i++){
                        w.at<float>(0,0) = labelPointXYZ[5][i].x - joints_position[1].x;
                        w.at<float>(1,0) = labelPointXYZ[5][i].y - joints_position[1].y;
                        w.at<float>(2,0) = labelPointXYZ[5][i].z - joints_position[1].z;

                        Mat c1 = w.t()*v;
                        Mat c2 = v.t()*v;
                        float indivi_distance;
                        int outlier = 0;
                        if( c1.at<float>(0,0) <= 0){
                            //indivi_distance = Distance_XYZXYZRGB(labelPointXYZ[5][i], joints_position[1]);
                            outlier = 1;
                        }
                        else if (c2.at<float>(0,0) <= c1.at<float>(0,0)){
                            //indivi_distance = Distance_XYZPoint3d(labelPointXYZ[5][i],temp_joints_position[0][0]);
                            outlier = 1;
                        }
                        else{
                            float b = c1.at<float>(0,0)/c2.at<float>(0,0);
                            pcl::PointXYZ p_xyz;
                            p_xyz.x = joints_position[1].x + b*v.at<float>(0,0);
                            p_xyz.y = joints_position[1].y + b*v.at<float>(1,0);
                            p_xyz.z = joints_position[1].z + b*v.at<float>(2,0);
                            indivi_distance = Distance_2XYZ(labelPointXYZ[5][i], p_xyz);
                            sum += indivi_distance;
                        }


                        if(indivi_distance < 0.005)
                            inlier_count = inlier_count + 1 - outlier;
                    }

                }
                //other thumb bone:
                else{
                    Mat v = Mat::zeros(3,1,CV_32FC1);
                    Mat w = Mat::zeros(3,1,CV_32FC1);
                    v.at<float>(0,0) = temp_joints_position[0][bone_index].x - temp_joints_position[0][bone_index-1].x;
                    v.at<float>(1,0) = temp_joints_position[0][bone_index].y - temp_joints_position[0][bone_index-1].y;
                    v.at<float>(2,0) = temp_joints_position[0][bone_index].z - temp_joints_position[0][bone_index-1].z;

                    //calculate distance of all points in label 5 to the line
                    for(int i = 0; i < labelPointXYZ[5+bone_index].size(); i++){
                        w.at<float>(0,0) = labelPointXYZ[5+bone_index][i].x - temp_joints_position[0][bone_index-1].x;
                        w.at<float>(1,0) = labelPointXYZ[5+bone_index][i].y - temp_joints_position[0][bone_index-1].y;
                        w.at<float>(2,0) = labelPointXYZ[5+bone_index][i].z - temp_joints_position[0][bone_index-1].z;

                        Mat c1 = w.t()*v;
                        Mat c2 = v.t()*v;
                        float indivi_distance;
                        int outlier = 0;
                        if( c1.at<float>(0,0) <= 0){
                            //indivi_distance = Distance_XYZPoint3d(labelPointXYZ[5+bone_index][i], temp_joints_position[0][bone_index-1]);
                            outlier = 1;
                        }
                        else if (c2.at<float>(0,0) <= c1.at<float>(0,0)){
                            //indivi_distance = Distance_XYZPoint3d(labelPointXYZ[5+bone_index][i],temp_joints_position[0][bone_index]);
                            outlier = 1;
                        }
                        else{
                            float b = c1.at<float>(0,0)/c2.at<float>(0,0);
                            pcl::PointXYZ p_xyz;
                            p_xyz.x = temp_joints_position[0][bone_index-1].x + b*v.at<float>(0,0);
                            p_xyz.y = temp_joints_position[0][bone_index-1].y + b*v.at<float>(1,0);
                            p_xyz.z = temp_joints_position[0][bone_index-1].z + b*v.at<float>(2,0);
                            indivi_distance = Distance_2XYZ(labelPointXYZ[5+bone_index][i], p_xyz);
                            sum += indivi_distance;

                        }


                        if(indivi_distance < 0.005)
                            inlier_count = inlier_count + 1 - outlier;
                    }
                }
            }
            float inlier_ratio = inlier_count*1.0/(labelPointXYZ[5].size() + labelPointXYZ[6].size() + labelPointXYZ[7].size());
            //std::cout << "inlier ratio: " << inlier_ratio << "    Average dis: " << sum/inlier_count << std::endl;
            if(inlier_ratio > T_ratio && sum/inlier_count < T_sum){
                joints_position[2].x = temp_joints_position[0][0].x;
                joints_position[2].y = temp_joints_position[0][0].y;
                joints_position[2].z = temp_joints_position[0][0].z;

                joints_position[3].x = temp_joints_position[0][1].x;
                joints_position[3].y = temp_joints_position[0][1].y;
                joints_position[3].z = temp_joints_position[0][1].z;

                joints_position[4].x = temp_joints_position[0][2].x;
                joints_position[4].y = temp_joints_position[0][2].y;
                joints_position[4].z = temp_joints_position[0][2].z;

                joints_position[5] = joints_position[4];
                T_ratio = inlier_ratio;
                T_sum = sum/inlier_count;
            }

        }
    }
    else{
        //2. Fitting 3 lines together for other fingers:
        for(int finger_index = finger2fit-1; finger_index < finger2fit; ++ finger_index){
            T_ratio = 0.0;
            T_sum = 1000;
            for(int iteration = 0; iteration < iteration_number; iteration++){
                //2.1 randomly select two points in intersection areas:
                Point3d temp_joints_position[5][3];
                temp_joints_position[finger_index][0].x = center[2*finger_index].x+rand()%5/1000;
                temp_joints_position[finger_index][0].y = center[2*finger_index].y+rand()%5/1000;
                temp_joints_position[finger_index][0].z = center[2*finger_index].z+rand()%5/1000;

                temp_joints_position[finger_index][1].x = center[1+2*finger_index].x+rand()%5/1000;
                temp_joints_position[finger_index][1].y = center[1+2*finger_index].y+rand()%5/1000;
                temp_joints_position[finger_index][1].z = center[1+2*finger_index].z+rand()%5/1000;

                //2.2 randomly select one point in end of distal areas:
                float distance = 0.0;
                int count = 0;
                while(count < 7){
                    int index = rand()% int( Distal_edge[finger_index].size() );
                    count ++;
                    float temp_distance = Distance_2Point3d(Distal_edge[finger_index][index], temp_joints_position[finger_index][1]);
                    if(temp_distance > distance){
                        distance = temp_distance;
                        temp_joints_position[finger_index][2] = Distal_edge[finger_index][index];
                    }
                }

                //2.3calculate the distance as score:
                float sum = 0;
                int inlier_count = 0;
                for(int bone_index = 0; bone_index < 3; ++bone_index){
                    //first finger bone:
                    if( bone_index  == 0){

                        Mat v = Mat::zeros(3,1,CV_32FC1);
                        Mat w = Mat::zeros(3,1,CV_32FC1);
                        v.at<float>(0,0) = temp_joints_position[finger_index][0].x - joints_position[5*finger_index+2].x;
                        v.at<float>(1,0) = temp_joints_position[finger_index][0].y - joints_position[5*finger_index+2].y;
                        v.at<float>(2,0) = temp_joints_position[finger_index][0].z - joints_position[5*finger_index+2].z;

                        //calculate distance of all points in label 5 to the line
                        for(int i = 0; i < labelPointXYZ[3*finger_index+5].size(); i++){
                            w.at<float>(0,0) = labelPointXYZ[3*finger_index+5][i].x - joints_position[5*finger_index+2].x;
                            w.at<float>(1,0) = labelPointXYZ[3*finger_index+5][i].y - joints_position[5*finger_index+2].y;
                            w.at<float>(2,0) = labelPointXYZ[3*finger_index+5][i].z - joints_position[5*finger_index+2].z;

                            Mat c1 = w.t()*v;
                            Mat c2 = v.t()*v;
                            float indivi_distance;
                            int outlier = 0;
                            if( c1.at<float>(0,0) <= 0){
                                //indivi_distance = Distance_XYZXYZRGB(labelPointXYZ[3*finger_index+5][i], joints_position[5*finger_index+2]);
                                outlier = 1;
                            }
                            else if (c2.at<float>(0,0) <= c1.at<float>(0,0)){
                                //indivi_distance = Distance_XYZPoint3d(labelPointXYZ[3*finger_index+5][i],temp_joints_position[finger_index][0]);
                                outlier = 1;
                            }
                            else{
                                float b = c1.at<float>(0,0)/c2.at<float>(0,0);
                                pcl::PointXYZ p_xyz;
                                p_xyz.x = joints_position[5*finger_index+2].x + b*v.at<float>(0,0);
                                p_xyz.y = joints_position[5*finger_index+2].y + b*v.at<float>(1,0);
                                p_xyz.z = joints_position[5*finger_index+2].z + b*v.at<float>(2,0);
                                indivi_distance = Distance_2XYZ(labelPointXYZ[3*finger_index+5][i], p_xyz);
                                sum += indivi_distance;
                            }

                            if(indivi_distance < 0.004)
                                inlier_count = inlier_count + 1 - outlier;
                        }

                    }
                    //other finger bone:
                    else{
                        Mat v = Mat::zeros(3,1,CV_32FC1);
                        Mat w = Mat::zeros(3,1,CV_32FC1);
                        v.at<float>(0,0) = temp_joints_position[finger_index][bone_index].x - temp_joints_position[finger_index][bone_index-1].x;
                        v.at<float>(1,0) = temp_joints_position[finger_index][bone_index].y - temp_joints_position[finger_index][bone_index-1].y;
                        v.at<float>(2,0) = temp_joints_position[finger_index][bone_index].z - temp_joints_position[finger_index][bone_index-1].z;

                        //calculate distance of all points in label 5 to the line
                        for(int i = 0; i < labelPointXYZ[3*finger_index+5+bone_index].size(); i++){
                            w.at<float>(0,0) = labelPointXYZ[3*finger_index+5+bone_index][i].x - temp_joints_position[finger_index][bone_index-1].x;
                            w.at<float>(1,0) = labelPointXYZ[3*finger_index+5+bone_index][i].y - temp_joints_position[finger_index][bone_index-1].y;
                            w.at<float>(2,0) = labelPointXYZ[3*finger_index+5+bone_index][i].z - temp_joints_position[finger_index][bone_index-1].z;

                            Mat c1 = w.t()*v;
                            Mat c2 = v.t()*v;
                            float indivi_distance;
                            int outlier = 0;
                            if( c1.at<float>(0,0) <= 0){
                                //indivi_distance = Distance_XYZPoint3d(labelPointXYZ[3*finger_index+5+bone_index][i], temp_joints_position[finger_index][bone_index-1]);
                                outlier = 1;
                            }
                            else if (c2.at<float>(0,0) <= c1.at<float>(0,0)){
                                //indivi_distance = Distance_XYZPoint3d(labelPointXYZ[3*finger_index+5+bone_index][i],temp_joints_position[finger_index][bone_index]);
                                outlier = 1;
                            }
                            else{
                                float b = c1.at<float>(0,0)/c2.at<float>(0,0);
                                pcl::PointXYZ p_xyz;
                                p_xyz.x = temp_joints_position[finger_index][bone_index-1].x + b*v.at<float>(0,0);
                                p_xyz.y = temp_joints_position[finger_index][bone_index-1].y + b*v.at<float>(1,0);
                                p_xyz.z = temp_joints_position[finger_index][bone_index-1].z + b*v.at<float>(2,0);
                                indivi_distance = Distance_2XYZ(labelPointXYZ[3*finger_index+5+bone_index][i], p_xyz);
                                sum += indivi_distance;
                            }

                            if(indivi_distance < 0.003)
                                inlier_count = inlier_count + 1 - outlier;
                        }
                    }
                }
                float inlier_ratio = inlier_count*1.0/(labelPointXYZ[3*finger_index+5].size() + labelPointXYZ[3*finger_index+6].size() + labelPointXYZ[3*finger_index+7].size());
                //std::cout << "inlier ratio: " << inlier_ratio << "    Average dis: " << sum/inlier_count << std::endl;
                if(inlier_ratio > T_ratio && sum/inlier_count < T_sum){
                    joints_position[5*finger_index+3].x = temp_joints_position[finger_index][0].x;
                    joints_position[5*finger_index+3].y = temp_joints_position[finger_index][0].y;
                    joints_position[5*finger_index+3].z = temp_joints_position[finger_index][0].z;

                    joints_position[5*finger_index+4].x = temp_joints_position[finger_index][1].x;
                    joints_position[5*finger_index+4].y = temp_joints_position[finger_index][1].y;
                    joints_position[5*finger_index+4].z = temp_joints_position[finger_index][1].z;

                    joints_position[5*finger_index+5].x = temp_joints_position[finger_index][2].x;
                    joints_position[5*finger_index+5].y = temp_joints_position[finger_index][2].y;
                    joints_position[5*finger_index+5].z = temp_joints_position[finger_index][2].z;

                    T_ratio = inlier_ratio;
                    T_sum = sum/inlier_count;
                }

            }
        }
    }

    //    //2.1 fitting first two joints to the mean center;
    //    joints_position[2].x = center[0].x;
    //    joints_position[2].y = center[0].y;
    //    joints_position[2].z = center[0].z;

    //    joints_position[3].x = center[1].x;
    //    joints_position[3].y = center[1].y;
    //    joints_position[3].z = center[1].z;

    //    for(int i = 1; i < 5; i++){

    //        joints_position[3+5*i].x = center[i*2].x;
    //        joints_position[3+5*i].y = center[i*2].y;
    //        joints_position[3+5*i].z = center[i*2].z;

    //        joints_position[4+5*i].x = center[i*2+1].x;
    //        joints_position[4+5*i].y = center[i*2+1].y;
    //        joints_position[4+5*i].z = center[i*2+1].z;
    //    }
    //    //2.2 fitting last joints(finger tip) according to the pcl;






}

void articulate_HandModel_XYZRGB::finger_fitting7(Mat const Hand_DepthMat, Mat const LabelMat, int const resolution, int const iteration_number, const vector< vector<pcl::PointXYZ> > labelPointXYZ, vector<int> & failed){
    vector< vector <Point3d> > Intersection(10, vector<Point3d>() );
    vector< vector <Point3d> > Distal_edge(5, vector<Point3d>() );
    int imageSize = 300/resolution;

    std::cout << "iteration_number: " << iteration_number << std::endl;
    srand((unsigned)time(NULL));

    //1. Looking for intersection region
    for(int row = 0; row < LabelMat.rows; row++){
        for(int col = 0; col < LabelMat.cols; col++){
            //intersection area:
            if( LabelMat.at<unsigned char>(row, col) != 0 && LabelMat.at<unsigned char>(row, col) != 3 && LabelMat.at<unsigned char>(row, col)%3 == 0){
                int L = LabelMat.at<unsigned char>(row, col);
                if(LabelMat.at<unsigned char>(row, col) - LabelMat.at<unsigned char>(row +1, col) == 1 ||  LabelMat.at<unsigned char>(row, col) - LabelMat.at<unsigned char>(row-1, col) == 1||
                        LabelMat.at<unsigned char>(row, col) - LabelMat.at<unsigned char>(row, col + 1) == 1 || LabelMat.at<unsigned char>(row, col) - LabelMat.at<unsigned char>(row, col - 1) == 1){
                    Point3d p3d;
                    p3d.x = (col-imageSize/2.0)*resolution/1000.0;
                    p3d.y = (row-imageSize/2.0)*resolution/1000.0;
                    p3d.z = (Hand_DepthMat.at<unsigned char>(row, col)-imageSize/2.0)*resolution/1000.0;
                    Intersection[(L/3-1)*2-2].push_back(p3d);
                }
                else if (LabelMat.at<unsigned char>(row, col) - LabelMat.at<unsigned char>(row +1, col) == -1 ||  LabelMat.at<unsigned char>(row, col) - LabelMat.at<unsigned char>(row-1, col) == -1||
                         LabelMat.at<unsigned char>(row, col) - LabelMat.at<unsigned char>(row, col + 1) == -1 || LabelMat.at<unsigned char>(row, col) - LabelMat.at<unsigned char>(row, col - 1) == -1){
                    Point3d p3d;
                    p3d.x = (col-imageSize/2.0)*resolution/1000.0;
                    p3d.y = (row-imageSize/2.0)*resolution/1000.0;
                    p3d.z = (Hand_DepthMat.at<unsigned char>(row, col)-imageSize/2.0)*resolution/1000.0;
                    Intersection[(L/3-1)*2-1].push_back(p3d);

                }

            }
            //distal edge:
            else if ( LabelMat.at<unsigned char>(row, col) != 1 && LabelMat.at<unsigned char>(row, col) != 4 && LabelMat.at<unsigned char>(row, col)%3 == 1){
                int L = LabelMat.at<unsigned char>(row, col);
                if(LabelMat.at<unsigned char>(row +1, col) == 0 || LabelMat.at<unsigned char>(row -1, col) == 0
                        || LabelMat.at<unsigned char>(row, col + 1) == 0 || LabelMat.at<unsigned char>(row, col - 1) == 0){
                    Point3d p3d;
                    p3d.x = (col-imageSize/2.0)*resolution/1000.0;
                    p3d.y = (row-imageSize/2.0)*resolution/1000.0;
                    p3d.z = (Hand_DepthMat.at<unsigned char>(row, col)-imageSize/2.0)*resolution/1000.0;
                    Distal_edge[(L-1)/3-2].push_back(p3d);
                }
            }
        }

    }
    //    for(int i = 0; i < 10; i++)
    //        std::cout<< "Size of " << i << ": " << Intersection[i].size() << std::endl;
    //1.2 Ransac to find the intersection center:
    Point3d center[10];
    vector<int> invalid_index;
    for(int i = 0; i < 10; i++){
        if(Intersection[i].size()!=0)
            Ransac(Intersection[i], center[i], 10, 0.007);
        else{
            center[i].x = -999;
            invalid_index.push_back(i);
        }
        //std::cout<< "Center " << i << ": " << center[i] << std::endl;
    }

    //2. Fitting 3 lines together for thumb
    float T_ratio = 0.0, T_sum = 1000;
    if(center[0].x != -999 && center[1].x != -999 && Distal_edge[0].size() != 0){
        for(int iteration = 0; iteration < iteration_number; iteration++){
            //2.1 randomly select two points in intersection areas:
            Point3d temp_joints_position[5][3];

            temp_joints_position[0][0].x = center[0].x+rand()%5/1000;
            temp_joints_position[0][0].y = center[0].y+rand()%5/1000;
            temp_joints_position[0][0].z = center[0].z+rand()%5/1000;

            temp_joints_position[0][1].x = center[1].x+rand()%5/1000;
            temp_joints_position[0][1].y = center[1].y+rand()%5/1000;
            temp_joints_position[0][1].z = center[1].z+rand()%5/1000;

            //2.2 randomly select one point in end of distal areas:
            float distance = 0.0;
            int count = 0;
            while(count < 10){
                int index = rand()% int( Distal_edge[0].size() );
                count ++;
                float temp_distance = Distance_2Point3d(Distal_edge[0][index], temp_joints_position[0][1]);
                if(temp_distance > distance){
                    distance = temp_distance;
                    temp_joints_position[0][2] = Distal_edge[0][index];
                }
            }

            //2.3calculate the distance as score:
            float sum = 0;
            int inlier_count = 0;
            for(int bone_index = 0; bone_index < 3; ++bone_index){
                //first thumb bone:
                if( bone_index  == 0){

                    Mat v = Mat::zeros(3,1,CV_32FC1);
                    Mat w = Mat::zeros(3,1,CV_32FC1);
                    v.at<float>(0,0) = temp_joints_position[0][0].x - joints_position[1].x;
                    v.at<float>(1,0) = temp_joints_position[0][0].y - joints_position[1].y;
                    v.at<float>(2,0) = temp_joints_position[0][0].z - joints_position[1].z;

                    //calculate distance of all points in label 5 to the line
                    for(int i = 0; i < labelPointXYZ[5].size(); i++){
                        w.at<float>(0,0) = labelPointXYZ[5][i].x - joints_position[1].x;
                        w.at<float>(1,0) = labelPointXYZ[5][i].y - joints_position[1].y;
                        w.at<float>(2,0) = labelPointXYZ[5][i].z - joints_position[1].z;

                        Mat c1 = w.t()*v;
                        Mat c2 = v.t()*v;
                        float indivi_distance;
                        int outlier = 0;
                        if( c1.at<float>(0,0) <= 0){
                            //indivi_distance = Distance_XYZXYZRGB(labelPointXYZ[5][i], joints_position[1]);
                            outlier = 1;
                        }
                        else if (c2.at<float>(0,0) <= c1.at<float>(0,0)){
                            //indivi_distance = Distance_XYZPoint3d(labelPointXYZ[5][i],temp_joints_position[0][0]);
                            outlier = 1;
                        }
                        else{
                            float b = c1.at<float>(0,0)/c2.at<float>(0,0);
                            pcl::PointXYZ p_xyz;
                            p_xyz.x = joints_position[1].x + b*v.at<float>(0,0);
                            p_xyz.y = joints_position[1].y + b*v.at<float>(1,0);
                            p_xyz.z = joints_position[1].z + b*v.at<float>(2,0);
                            indivi_distance = Distance_2XYZ(labelPointXYZ[5][i], p_xyz);
                            sum += indivi_distance;
                        }


                        if(indivi_distance < 0.005)
                            inlier_count = inlier_count + 1 - outlier;
                    }

                }
                //other thumb bone:
                else{
                    Mat v = Mat::zeros(3,1,CV_32FC1);
                    Mat w = Mat::zeros(3,1,CV_32FC1);
                    v.at<float>(0,0) = temp_joints_position[0][bone_index].x - temp_joints_position[0][bone_index-1].x;
                    v.at<float>(1,0) = temp_joints_position[0][bone_index].y - temp_joints_position[0][bone_index-1].y;
                    v.at<float>(2,0) = temp_joints_position[0][bone_index].z - temp_joints_position[0][bone_index-1].z;

                    //calculate distance of all points in label 5 to the line
                    for(int i = 0; i < labelPointXYZ[5+bone_index].size(); i++){
                        w.at<float>(0,0) = labelPointXYZ[5+bone_index][i].x - temp_joints_position[0][bone_index-1].x;
                        w.at<float>(1,0) = labelPointXYZ[5+bone_index][i].y - temp_joints_position[0][bone_index-1].y;
                        w.at<float>(2,0) = labelPointXYZ[5+bone_index][i].z - temp_joints_position[0][bone_index-1].z;

                        Mat c1 = w.t()*v;
                        Mat c2 = v.t()*v;
                        float indivi_distance;
                        int outlier = 0;
                        if( c1.at<float>(0,0) <= 0){
                            //indivi_distance = Distance_XYZPoint3d(labelPointXYZ[5+bone_index][i], temp_joints_position[0][bone_index-1]);
                            outlier = 1;
                        }
                        else if (c2.at<float>(0,0) <= c1.at<float>(0,0)){
                            //indivi_distance = Distance_XYZPoint3d(labelPointXYZ[5+bone_index][i],temp_joints_position[0][bone_index]);
                            outlier = 1;
                        }
                        else{
                            float b = c1.at<float>(0,0)/c2.at<float>(0,0);
                            pcl::PointXYZ p_xyz;
                            p_xyz.x = temp_joints_position[0][bone_index-1].x + b*v.at<float>(0,0);
                            p_xyz.y = temp_joints_position[0][bone_index-1].y + b*v.at<float>(1,0);
                            p_xyz.z = temp_joints_position[0][bone_index-1].z + b*v.at<float>(2,0);
                            indivi_distance = Distance_2XYZ(labelPointXYZ[5+bone_index][i], p_xyz);
                            sum += indivi_distance;

                        }


                        if(indivi_distance < 0.005)
                            inlier_count = inlier_count + 1 - outlier;
                    }
                }
            }
            float inlier_ratio = inlier_count*1.0/(labelPointXYZ[5].size() + labelPointXYZ[6].size() + labelPointXYZ[7].size());
            //std::cout << "inlier ratio: " << inlier_ratio << "    Average dis: " << sum/inlier_count << std::endl;
            if(inlier_ratio > T_ratio && sum/inlier_count < T_sum){
                joints_position[2].x = temp_joints_position[0][0].x;
                joints_position[2].y = temp_joints_position[0][0].y;
                joints_position[2].z = temp_joints_position[0][0].z;

                joints_position[3].x = temp_joints_position[0][1].x;
                joints_position[3].y = temp_joints_position[0][1].y;
                joints_position[3].z = temp_joints_position[0][1].z;

                joints_position[4].x = temp_joints_position[0][2].x;
                joints_position[4].y = temp_joints_position[0][2].y;
                joints_position[4].z = temp_joints_position[0][2].z;

                joints_position[5] = joints_position[4];
                T_ratio = inlier_ratio;
                T_sum = sum/inlier_count;
            }

        }
    }
    else{
        failed.push_back(1);
    }


    //2. Fitting 3 lines together for other fingers:
    for(int finger_index = 1; finger_index < 5; ++ finger_index){
        if(center[2*finger_index].x != -999 && center[2*finger_index+1].x != -999 && Distal_edge[finger_index].size() != 0){
            T_ratio = 0.0;
            T_sum = 1000;
            for(int iteration = 0; iteration < iteration_number; iteration++){
                //2.1 randomly select two points in intersection areas:
                Point3d temp_joints_position[5][3];
                temp_joints_position[finger_index][0].x = center[2*finger_index].x+rand()%5/1000;
                temp_joints_position[finger_index][0].y = center[2*finger_index].y+rand()%5/1000;
                temp_joints_position[finger_index][0].z = center[2*finger_index].z+rand()%5/1000;

                temp_joints_position[finger_index][1].x = center[1+2*finger_index].x+rand()%5/1000;
                temp_joints_position[finger_index][1].y = center[1+2*finger_index].y+rand()%5/1000;
                temp_joints_position[finger_index][1].z = center[1+2*finger_index].z+rand()%5/1000;

                //2.2 randomly select one point in end of distal areas:
                float distance = 0.0;
                int count = 0;
                while(count < 7){
                    int index = rand()% int( Distal_edge[finger_index].size() );
                    count ++;
                    float temp_distance = Distance_2Point3d(Distal_edge[finger_index][index], temp_joints_position[finger_index][1]);
                    if(temp_distance > distance){
                        distance = temp_distance;
                        temp_joints_position[finger_index][2] = Distal_edge[finger_index][index];
                    }
                }

                //2.3calculate the distance as score:
                float sum = 0;
                int inlier_count = 0;
                for(int bone_index = 0; bone_index < 3; ++bone_index){
                    //first finger bone:
                    if( bone_index  == 0){

                        Mat v = Mat::zeros(3,1,CV_32FC1);
                        Mat w = Mat::zeros(3,1,CV_32FC1);
                        v.at<float>(0,0) = temp_joints_position[finger_index][0].x - joints_position[5*finger_index+2].x;
                        v.at<float>(1,0) = temp_joints_position[finger_index][0].y - joints_position[5*finger_index+2].y;
                        v.at<float>(2,0) = temp_joints_position[finger_index][0].z - joints_position[5*finger_index+2].z;

                        //calculate distance of all points in label 5 to the line
                        for(int i = 0; i < labelPointXYZ[3*finger_index+5].size(); i++){
                            w.at<float>(0,0) = labelPointXYZ[3*finger_index+5][i].x - joints_position[5*finger_index+2].x;
                            w.at<float>(1,0) = labelPointXYZ[3*finger_index+5][i].y - joints_position[5*finger_index+2].y;
                            w.at<float>(2,0) = labelPointXYZ[3*finger_index+5][i].z - joints_position[5*finger_index+2].z;

                            Mat c1 = w.t()*v;
                            Mat c2 = v.t()*v;
                            float indivi_distance;
                            int outlier = 0;
                            if( c1.at<float>(0,0) <= 0){
                                //indivi_distance = Distance_XYZXYZRGB(labelPointXYZ[3*finger_index+5][i], joints_position[5*finger_index+2]);
                                outlier = 1;
                            }
                            else if (c2.at<float>(0,0) <= c1.at<float>(0,0)){
                                //indivi_distance = Distance_XYZPoint3d(labelPointXYZ[3*finger_index+5][i],temp_joints_position[finger_index][0]);
                                outlier = 1;
                            }
                            else{
                                float b = c1.at<float>(0,0)/c2.at<float>(0,0);
                                pcl::PointXYZ p_xyz;
                                p_xyz.x = joints_position[5*finger_index+2].x + b*v.at<float>(0,0);
                                p_xyz.y = joints_position[5*finger_index+2].y + b*v.at<float>(1,0);
                                p_xyz.z = joints_position[5*finger_index+2].z + b*v.at<float>(2,0);
                                indivi_distance = Distance_2XYZ(labelPointXYZ[3*finger_index+5][i], p_xyz);
                                sum += indivi_distance;
                            }

                            if(indivi_distance < 0.004)
                                inlier_count = inlier_count + 1 - outlier;
                        }

                    }
                    //other finger bone:
                    else{
                        Mat v = Mat::zeros(3,1,CV_32FC1);
                        Mat w = Mat::zeros(3,1,CV_32FC1);
                        v.at<float>(0,0) = temp_joints_position[finger_index][bone_index].x - temp_joints_position[finger_index][bone_index-1].x;
                        v.at<float>(1,0) = temp_joints_position[finger_index][bone_index].y - temp_joints_position[finger_index][bone_index-1].y;
                        v.at<float>(2,0) = temp_joints_position[finger_index][bone_index].z - temp_joints_position[finger_index][bone_index-1].z;

                        //calculate distance of all points in label 5 to the line
                        for(int i = 0; i < labelPointXYZ[3*finger_index+5+bone_index].size(); i++){
                            w.at<float>(0,0) = labelPointXYZ[3*finger_index+5+bone_index][i].x - temp_joints_position[finger_index][bone_index-1].x;
                            w.at<float>(1,0) = labelPointXYZ[3*finger_index+5+bone_index][i].y - temp_joints_position[finger_index][bone_index-1].y;
                            w.at<float>(2,0) = labelPointXYZ[3*finger_index+5+bone_index][i].z - temp_joints_position[finger_index][bone_index-1].z;

                            Mat c1 = w.t()*v;
                            Mat c2 = v.t()*v;
                            float indivi_distance;
                            int outlier = 0;
                            if( c1.at<float>(0,0) <= 0){
                                //indivi_distance = Distance_XYZPoint3d(labelPointXYZ[3*finger_index+5+bone_index][i], temp_joints_position[finger_index][bone_index-1]);
                                outlier = 1;
                            }
                            else if (c2.at<float>(0,0) <= c1.at<float>(0,0)){
                                //indivi_distance = Distance_XYZPoint3d(labelPointXYZ[3*finger_index+5+bone_index][i],temp_joints_position[finger_index][bone_index]);
                                outlier = 1;
                            }
                            else{
                                float b = c1.at<float>(0,0)/c2.at<float>(0,0);
                                pcl::PointXYZ p_xyz;
                                p_xyz.x = temp_joints_position[finger_index][bone_index-1].x + b*v.at<float>(0,0);
                                p_xyz.y = temp_joints_position[finger_index][bone_index-1].y + b*v.at<float>(1,0);
                                p_xyz.z = temp_joints_position[finger_index][bone_index-1].z + b*v.at<float>(2,0);
                                indivi_distance = Distance_2XYZ(labelPointXYZ[3*finger_index+5+bone_index][i], p_xyz);
                                sum += indivi_distance;
                            }

                            if(indivi_distance < 0.003)
                                inlier_count = inlier_count + 1 - outlier;
                        }
                    }
                }
                float inlier_ratio = inlier_count*1.0/(labelPointXYZ[3*finger_index+5].size() + labelPointXYZ[3*finger_index+6].size() + labelPointXYZ[3*finger_index+7].size());
                //std::cout << "inlier ratio: " << inlier_ratio << "    Average dis: " << sum/inlier_count << std::endl;
                if(inlier_ratio > T_ratio && sum/inlier_count < T_sum){
                    joints_position[5*finger_index+3].x = temp_joints_position[finger_index][0].x;
                    joints_position[5*finger_index+3].y = temp_joints_position[finger_index][0].y;
                    joints_position[5*finger_index+3].z = temp_joints_position[finger_index][0].z;

                    joints_position[5*finger_index+4].x = temp_joints_position[finger_index][1].x;
                    joints_position[5*finger_index+4].y = temp_joints_position[finger_index][1].y;
                    joints_position[5*finger_index+4].z = temp_joints_position[finger_index][1].z;

                    joints_position[5*finger_index+5].x = temp_joints_position[finger_index][2].x;
                    joints_position[5*finger_index+5].y = temp_joints_position[finger_index][2].y;
                    joints_position[5*finger_index+5].z = temp_joints_position[finger_index][2].z;

                    T_ratio = inlier_ratio;
                    T_sum = sum/inlier_count;
                }

            }
        }
        else{
            failed.push_back(finger_index);
        }
    }
}

void articulate_HandModel_XYZRGB::bp_finger_fitting(Mat const Hand_DepthMat, Mat const LabelMat, int const resolution, int const iteration_number, const vector< vector<pcl::PointXYZ> > labelPointXYZ, vector<int> & failed){

}

void articulate_HandModel_XYZRGB::bp_kinematic_constrain(const vector<int> failed, vector<Point3d> &newpoint){
    Point3d w[5];
    bool complete_fingers[5] = {true, true, true, true, true};
    for(int i = 0; i< failed.size(); i++){
        complete_fingers[failed[i]] = false;
    }
    //thumb:
    Mat translation = Mat::zeros(3,1,CV_32FC1);
    translation.at<float>(0,0) = parameters[0];
    translation.at<float>(1,0) = parameters[1];
    translation.at<float>(2,0) = parameters[2];

    if(complete_fingers[0] == true){
        Mat joints_for_calc = Mat::zeros(3,1,CV_32FC1);
        virtual_joints[0].copyTo(joints_for_calc);

        Mat R_p_r_y = R_z(parameters[5])*R_y(parameters[4])*R_x(parameters[3]);


        joints_for_calc = R_p_r_y * joints_for_calc+translation;

        Point3d temp_w[3];
        for(int i = 0; i<3; i++){
            float dx1 = joints_for_calc.at<float>(0,0) - joints_position[1].x;
            float dy1 = joints_for_calc.at<float>(1,0) - joints_position[1].y;
            float dz1 = joints_for_calc.at<float>(2,0) - joints_position[1].z;

            float dx2 = joints_position[2+i].x - joints_position[1].x;
            float dy2 = joints_position[2+i].y - joints_position[1].y;
            float dz2 = joints_position[2+i].z - joints_position[1].z;

            temp_w[i].x = 1;
            if(dy1*dz2-dy2*dz1 == 0){
                temp_w[i].y = 100000;
                temp_w[i].z = 100000;
            }
            else{
                temp_w[i].y = (dx2*dz1-dx1*dz2)/(dy1*dz2-dy2*dz1);
                temp_w[i].z = (dx2*dy1-dx1*dy2)/(dy2*dz1-dy1*dz2);
            }

            float length = sqrt(temp_w[i].x * temp_w[i].x + temp_w[i].y * temp_w[i].y + temp_w[i].z * temp_w[i].z);
            if(length!= 0){
                temp_w[i].x = temp_w[i].x/length;
                temp_w[i].y = temp_w[i].y/length;
                temp_w[i].z = temp_w[i].z/length;
            }
        }


        w[0].x = (temp_w[0].x*Distance_2XYZRGB( joints_position[2],  joints_position[1]) +
                  temp_w[1].x*Distance_2XYZRGB( joints_position[3],  joints_position[1])+
                  temp_w[2].x*Distance_2XYZRGB( joints_position[4],  joints_position[1]))/
                (Distance_2XYZRGB( joints_position[2],  joints_position[1])+Distance_2XYZRGB( joints_position[3],  joints_position[1])+Distance_2XYZRGB( joints_position[4],  joints_position[1]));
        w[0].y = (temp_w[0].y*Distance_2XYZRGB( joints_position[2],  joints_position[1]) +
                  temp_w[1].y*Distance_2XYZRGB( joints_position[3],  joints_position[1])+
                  temp_w[2].y*Distance_2XYZRGB( joints_position[4],  joints_position[1]))/
                (Distance_2XYZRGB( joints_position[1],  joints_position[1])+Distance_2XYZRGB( joints_position[3],  joints_position[1])+Distance_2XYZRGB( joints_position[4],  joints_position[1]));
        w[0].z = (temp_w[0].z*Distance_2XYZRGB( joints_position[2],  joints_position[1]) +
                  temp_w[1].z*Distance_2XYZRGB( joints_position[3],  joints_position[1])+
                  temp_w[2].z*Distance_2XYZRGB( joints_position[4],  joints_position[1]))/
                (Distance_2XYZRGB( joints_position[2],  joints_position[1])+Distance_2XYZRGB( joints_position[3],  joints_position[1])+Distance_2XYZRGB( joints_position[4],  joints_position[1]));


        //        for(int b = 0; b<3; b++){
        //            Point3d v;
        //            v.x = joints_position[2+b].x - joints_position[1].x;
        //            v.y = joints_position[2+b].y - joints_position[1].y;
        //            v.z = joints_position[2+b].z - joints_position[1].z;
        //            float dist = v.x*w[0].x + v.y*w[0].y + v.z*w[0].z;
        //            joints_position[2+b].x = joints_position[2+b].x - dist*w[0].x;
        //            joints_position[2+b].y = joints_position[2+b].y - dist*w[0].y;
        //            joints_position[2+b].z = joints_position[2+b].z - dist*w[0].z;

        //            float scale = bone_length[0][b]/1000.0/Distance_2XYZRGB(joints_position[2+b], joints_position[1+b]);
        //            joints_position[2+b].x = (joints_position[2+b].x - joints_position[1+b].x)*scale + joints_position[1+b].x;
        //            joints_position[2+b].y = (joints_position[2+b].y - joints_position[1+b].y)*scale + joints_position[1+b].y;
        //            joints_position[2+b].z = (joints_position[2+b].z - joints_position[1+b].z)*scale + joints_position[1+b].z;


        //        }
        for(int b = 0; b<3; b++){
            Point3d v;
            v.x = joints_position[3-b].x - joints_position[4].x;
            v.y = joints_position[3-b].y - joints_position[4].y;
            v.z = joints_position[3-b].z - joints_position[4].z;
            float dist = v.x*w[0].x + v.y*w[0].y + v.z*w[0].z;

            joints_position[3-b].x = joints_position[3-b].x - dist*w[0].x;
            joints_position[3-b].y = joints_position[3-b].y - dist*w[0].y;
            joints_position[3-b].z = joints_position[3-b].z - dist*w[0].z;

            float scale = bone_length[0][2-b]/1000.0/Distance_2XYZRGB(joints_position[3-b], joints_position[4-b]);
            joints_position[3-b].x = (joints_position[3-b].x - joints_position[4-b].x)*scale + joints_position[4-b].x;
            joints_position[3-b].y = (joints_position[3-b].y - joints_position[4-b].y)*scale + joints_position[4-b].y;
            joints_position[3-b].z = (joints_position[3-b].z - joints_position[4-b].z)*scale + joints_position[4-b].z;

        }

        joints_position[5] = joints_position[4];
    }

    //other fingers:
    for(int f = 1; f < 5; f++){
        if(complete_fingers[f] == true){
            Mat joints_for_calc = Mat::zeros(3,1,CV_32FC1);
            virtual_joints[f].copyTo(joints_for_calc);
            //calculate the point directly over palm end joint:
            Mat R_p_r_y = R_z(parameters[5])*R_y(parameters[4])*R_x(parameters[3]);

            //std::cout << "R: " << R_p_r_y << std::endl;


            //std::cout << "translation: " << translation << std::endl;
            joints_for_calc = R_p_r_y * joints_for_calc+translation;
            //calculate plane:
            Point3d temp_w[3];
            for(int i = 0; i<3; i++){
                float dx1 = joints_for_calc.at<float>(0,0) - joints_position[5*f+2].x;
                float dy1 = joints_for_calc.at<float>(1,0) - joints_position[5*f+2].y;
                float dz1 = joints_for_calc.at<float>(2,0) - joints_position[5*f+2].z;

                float dx2 = joints_position[5*f+3+i].x - joints_position[5*f+2].x;
                float dy2 = joints_position[5*f+3+i].y - joints_position[5*f+2].y;
                float dz2 = joints_position[5*f+3+i].z - joints_position[5*f+2].z;

                temp_w[i].x = 1;
                if(dy1*dz2-dy2*dz1 == 0){
                    temp_w[i].y = 100000;
                    temp_w[i].z = 100000;
                }
                else{
                    temp_w[i].y = (dx2*dz1-dx1*dz2)/(dy1*dz2-dy2*dz1);
                    temp_w[i].z = (dx2*dy1-dx1*dy2)/(dy2*dz1-dy1*dz2);
                }

                float length = sqrt(temp_w[i].x * temp_w[i].x + temp_w[i].y * temp_w[i].y + temp_w[i].z * temp_w[i].z);
                if(length!= 0){
                    temp_w[i].x = temp_w[i].x/length;
                    temp_w[i].y = temp_w[i].y/length;
                    temp_w[i].z = temp_w[i].z/length;
                }
            }
            //plane: a = w[f].x; b = w[f].y; c = w[f].z;

            w[f].x = (temp_w[0].x*Distance_2XYZRGB( joints_position[5*f+3],  joints_position[5*f+2]) +
                      temp_w[1].x*Distance_2XYZRGB( joints_position[5*f+4],  joints_position[5*f+2])+
                      temp_w[2].x*Distance_2XYZRGB( joints_position[5*f+5],  joints_position[5*f+2]))/
                    (Distance_2XYZRGB( joints_position[5*f+3],  joints_position[5*f+2])+Distance_2XYZRGB( joints_position[5*f+4],  joints_position[5*f+2])+Distance_2XYZRGB( joints_position[5*f+5],  joints_position[5*f+2]));
            w[f].y = (temp_w[0].y*Distance_2XYZRGB( joints_position[5*f+3],  joints_position[5*f+2]) +
                      temp_w[1].y*Distance_2XYZRGB( joints_position[5*f+4],  joints_position[5*f+2])+
                      temp_w[2].y*Distance_2XYZRGB( joints_position[5*f+5],  joints_position[5*f+2]))/
                    (Distance_2XYZRGB( joints_position[5*f+3],  joints_position[5*f+2])+Distance_2XYZRGB( joints_position[5*f+4],  joints_position[5*f+2])+Distance_2XYZRGB( joints_position[5*f+5],  joints_position[5*f+2]));
            w[f].z = (temp_w[0].z*Distance_2XYZRGB( joints_position[5*f+3],  joints_position[5*f+2]) +
                      temp_w[1].z*Distance_2XYZRGB( joints_position[5*f+4],  joints_position[5*f+2])+
                      temp_w[2].z*Distance_2XYZRGB( joints_position[5*f+5],  joints_position[5*f+2]))/
                    (Distance_2XYZRGB( joints_position[5*f+3],  joints_position[5*f+2])+Distance_2XYZRGB( joints_position[5*f+4],  joints_position[5*f+2])+Distance_2XYZRGB( joints_position[5*f+5],  joints_position[5*f+2]));

            std::cout << f << ": " << w[f] << std::endl;

            //            for(int b = 0; b<3; b++){
            //                Point3d v;
            //                v.x = joints_position[5*f+3+b].x - joints_position[5*f+2].x;
            //                v.y = joints_position[5*f+3+b].y - joints_position[5*f+2].y;
            //                v.z = joints_position[5*f+3+b].z - joints_position[5*f+2].z;
            //                float dist = v.x*w[f].x + v.y*w[f].y + v.z*w[f].z;
            //                joints_position[5*f+3+b].x = joints_position[5*f+3+b].x - dist*w[f].x;
            //                joints_position[5*f+3+b].y = joints_position[5*f+3+b].y - dist*w[f].y;
            //                joints_position[5*f+3+b].z = joints_position[5*f+3+b].z - dist*w[f].z;

            //                float scale = bone_length[f][b+1]/1000.0/Distance_2XYZRGB(joints_position[5*f+3+b], joints_position[5*f+2+b]);
            //                joints_position[5*f+3+b].x = (joints_position[5*f+3+b].x - joints_position[5*f+2+b].x)*scale + joints_position[5*f+2+b].x;
            //                joints_position[5*f+3+b].y = (joints_position[5*f+3+b].y - joints_position[5*f+2+b].y)*scale + joints_position[5*f+2+b].y;
            //                joints_position[5*f+3+b].z = (joints_position[5*f+3+b].z - joints_position[5*f+2+b].z)*scale + joints_position[5*f+2+b].z;

            //                //            if(abs(Distance_2XYZRGB(joints_position[5*f+3+b], joints_position[5*f+2+b]) - bone_length[f][b+1]/1000.0)>0.001)
            //                //                std::cout << "No!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;


            //            }
            for(int b = 0; b<3; b++){
                Point3d v;
                v.x = joints_position[5*f+4-b].x - joints_position[5*f+5].x;
                v.y = joints_position[5*f+4-b].y - joints_position[5*f+5].y;
                v.z = joints_position[5*f+4-b].z - joints_position[5*f+5].z;
                float dist = v.x*w[f].x + v.y*w[f].y + v.z*w[f].z;
                joints_position[5*f+4-b].x = joints_position[5*f+4-b].x - dist*w[f].x;
                joints_position[5*f+4-b].y = joints_position[5*f+4-b].y - dist*w[f].y;
                joints_position[5*f+4-b].z = joints_position[5*f+4-b].z - dist*w[f].z;

                float scale = bone_length[f][3-b]/1000.0/Distance_2XYZRGB(joints_position[5*f+4-b], joints_position[5*f+5-b]);
                joints_position[5*f+4-b].x = (joints_position[5*f+4-b].x - joints_position[5*f+5-b].x)*scale + joints_position[5*f+5-b].x;
                joints_position[5*f+4-b].y = (joints_position[5*f+4-b].y - joints_position[5*f+5-b].y)*scale + joints_position[5*f+5-b].y;
                joints_position[5*f+4-b].z = (joints_position[5*f+4-b].z - joints_position[5*f+5-b].z)*scale + joints_position[5*f+5-b].z;

                //            if(abs(Distance_2XYZRGB(joints_position[5*f+3+b], joints_position[5*f+2+b]) - bone_length[f][b+1]/1000.0)>0.001)
                //                std::cout << "No!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;


            }



        }
    }
    ROS_INFO("back propagation");
}

void articulate_HandModel_XYZRGB::constrain_based_smooth(int number_of_joints){
    Point3d w[5];

    //thumb:
    Mat joints_for_calc = Mat::zeros(3,1,CV_32FC1);
    virtual_joints[0].copyTo(joints_for_calc);

    Mat R_p_r_y = R_z(parameters[5])*R_y(parameters[4])*R_x(parameters[3]);
    Mat translation = Mat::zeros(3,1,CV_32FC1);
    translation.at<float>(0,0) = parameters[0];
    translation.at<float>(1,0) = parameters[1];
    translation.at<float>(2,0) = parameters[2];

    joints_for_calc = R_p_r_y * joints_for_calc+translation;

    Point3d temp_w[number_of_joints];
    for(int i = 0; i<number_of_joints; i++){
        float dx1 = joints_for_calc.at<float>(0,0) - joints_position[1].x;
        float dy1 = joints_for_calc.at<float>(1,0) - joints_position[1].y;
        float dz1 = joints_for_calc.at<float>(2,0) - joints_position[1].z;

        float dx2 = joints_position[2+i].x - joints_position[1].x;
        float dy2 = joints_position[2+i].y - joints_position[1].y;
        float dz2 = joints_position[2+i].z - joints_position[1].z;

        temp_w[i].x = 1;
        if(dy1*dz2-dy2*dz1 == 0){
            temp_w[i].y = 100000;
            temp_w[i].z = 100000;
        }
        else{
            temp_w[i].y = (dx2*dz1-dx1*dz2)/(dy1*dz2-dy2*dz1);
            temp_w[i].z = (dx2*dy1-dx1*dy2)/(dy2*dz1-dy1*dz2);
        }

        float length = sqrt(temp_w[i].x * temp_w[i].x + temp_w[i].y * temp_w[i].y + temp_w[i].z * temp_w[i].z);
        if(length!= 0){
            temp_w[i].x = temp_w[i].x/length;
            temp_w[i].y = temp_w[i].y/length;
            temp_w[i].z = temp_w[i].z/length;
        }
    }

    if(number_of_joints == 3){
        w[0].x = (temp_w[0].x*Distance_2XYZRGB( joints_position[2],  joints_position[1]) +
                  temp_w[1].x*Distance_2XYZRGB( joints_position[3],  joints_position[1])+
                  temp_w[2].x*Distance_2XYZRGB( joints_position[4],  joints_position[1]))/
                (Distance_2XYZRGB( joints_position[2],  joints_position[1])+Distance_2XYZRGB( joints_position[3],  joints_position[1])+Distance_2XYZRGB( joints_position[4],  joints_position[1]));
        w[0].y = (temp_w[0].y*Distance_2XYZRGB( joints_position[2],  joints_position[1]) +
                  temp_w[1].y*Distance_2XYZRGB( joints_position[3],  joints_position[1])+
                  temp_w[2].y*Distance_2XYZRGB( joints_position[4],  joints_position[1]))/
                (Distance_2XYZRGB( joints_position[1],  joints_position[1])+Distance_2XYZRGB( joints_position[3],  joints_position[1])+Distance_2XYZRGB( joints_position[4],  joints_position[1]));
        w[0].z = (temp_w[0].z*Distance_2XYZRGB( joints_position[2],  joints_position[1]) +
                  temp_w[1].z*Distance_2XYZRGB( joints_position[3],  joints_position[1])+
                  temp_w[2].z*Distance_2XYZRGB( joints_position[4],  joints_position[1]))/
                (Distance_2XYZRGB( joints_position[2],  joints_position[1])+Distance_2XYZRGB( joints_position[3],  joints_position[1])+Distance_2XYZRGB( joints_position[4],  joints_position[1]));
    }
    else if (number_of_joints == 2){
        w[0].x = (temp_w[0].x*Distance_2XYZRGB( joints_position[2],  joints_position[1]) +
                  temp_w[1].x*Distance_2XYZRGB( joints_position[3],  joints_position[1]))/
                (Distance_2XYZRGB( joints_position[2],  joints_position[1])+Distance_2XYZRGB( joints_position[3],  joints_position[1]));
        w[0].y = (temp_w[0].y*Distance_2XYZRGB( joints_position[2],  joints_position[1]) +
                  temp_w[1].y*Distance_2XYZRGB( joints_position[3],  joints_position[1]))/
                (Distance_2XYZRGB( joints_position[2],  joints_position[1])+Distance_2XYZRGB( joints_position[3],  joints_position[1]));
        w[0].z = (temp_w[0].z*Distance_2XYZRGB( joints_position[2],  joints_position[1]) +
                  temp_w[1].z*Distance_2XYZRGB( joints_position[3],  joints_position[1]))/
                (Distance_2XYZRGB( joints_position[2],  joints_position[1])+Distance_2XYZRGB( joints_position[3],  joints_position[1]));
    }
    else{
        w[0] = temp_w[0];
    }

    for(int b = 0; b<number_of_joints; b++){
        Point3d v;
        v.x = joints_position[2+b].x - joints_position[1].x;
        v.y = joints_position[2+b].y - joints_position[1].y;
        v.z = joints_position[2+b].z - joints_position[1].z;
        float dist = v.x*w[0].x + v.y*w[0].y + v.z*w[0].z;
        joints_position[2+b].x = joints_position[2+b].x - dist*w[0].x;
        joints_position[2+b].y = joints_position[2+b].y - dist*w[0].y;
        joints_position[2+b].z = joints_position[2+b].z - dist*w[0].z;

        float scale = bone_length[0][b]/1000.0/Distance_2XYZRGB(joints_position[2+b], joints_position[1+b]);
        joints_position[2+b].x = (joints_position[2+b].x - joints_position[1+b].x)*scale + joints_position[1+b].x;
        joints_position[2+b].y = (joints_position[2+b].y - joints_position[1+b].y)*scale + joints_position[1+b].y;
        joints_position[2+b].z = (joints_position[2+b].z - joints_position[1+b].z)*scale + joints_position[1+b].z;


    }
    joints_position[5] = joints_position[4];


    //other fingers:
    for(int f = 1; f < 5; f++){
        joints_for_calc = Mat::zeros(3,1,CV_32FC1);
        virtual_joints[f].copyTo(joints_for_calc);
        //calculate the point directly over palm end joint:
        R_p_r_y = R_z(parameters[5])*R_y(parameters[4])*R_x(parameters[3]);

        //std::cout << "R: " << R_p_r_y << std::endl;


        //std::cout << "translation: " << translation << std::endl;
        joints_for_calc = R_p_r_y * joints_for_calc+translation;
        //calculate plane:
        temp_w[number_of_joints];
        for(int i = 0; i<number_of_joints; i++){
            float dx1 = joints_for_calc.at<float>(0,0) - joints_position[5*f+2].x;
            float dy1 = joints_for_calc.at<float>(1,0) - joints_position[5*f+2].y;
            float dz1 = joints_for_calc.at<float>(2,0) - joints_position[5*f+2].z;

            float dx2 = joints_position[5*f+3+i].x - joints_position[5*f+2].x;
            float dy2 = joints_position[5*f+3+i].y - joints_position[5*f+2].y;
            float dz2 = joints_position[5*f+3+i].z - joints_position[5*f+2].z;

            temp_w[i].x = 1;
            if(dy1*dz2-dy2*dz1 == 0){
                temp_w[i].y = 100000;
                temp_w[i].z = 100000;
            }
            else{
                temp_w[i].y = (dx2*dz1-dx1*dz2)/(dy1*dz2-dy2*dz1);
                temp_w[i].z = (dx2*dy1-dx1*dy2)/(dy2*dz1-dy1*dz2);
            }

            float length = sqrt(temp_w[i].x * temp_w[i].x + temp_w[i].y * temp_w[i].y + temp_w[i].z * temp_w[i].z);
            if(length!= 0){
                temp_w[i].x = temp_w[i].x/length;
                temp_w[i].y = temp_w[i].y/length;
                temp_w[i].z = temp_w[i].z/length;
            }
        }
        //plane: a = w[f].x; b = w[f].y; c = w[f].z;
        if(number_of_joints == 3){
            w[f].x = (temp_w[0].x*Distance_2XYZRGB( joints_position[5*f+3],  joints_position[5*f+2]) +
                      temp_w[1].x*Distance_2XYZRGB( joints_position[5*f+4],  joints_position[5*f+2])+
                      temp_w[2].x*Distance_2XYZRGB( joints_position[5*f+5],  joints_position[5*f+2]))/
                    (Distance_2XYZRGB( joints_position[5*f+3],  joints_position[5*f+2])+Distance_2XYZRGB( joints_position[5*f+4],  joints_position[5*f+2])+Distance_2XYZRGB( joints_position[5*f+5],  joints_position[5*f+2]));
            w[f].y = (temp_w[0].y*Distance_2XYZRGB( joints_position[5*f+3],  joints_position[5*f+2]) +
                      temp_w[1].y*Distance_2XYZRGB( joints_position[5*f+4],  joints_position[5*f+2])+
                      temp_w[2].y*Distance_2XYZRGB( joints_position[5*f+5],  joints_position[5*f+2]))/
                    (Distance_2XYZRGB( joints_position[5*f+3],  joints_position[5*f+2])+Distance_2XYZRGB( joints_position[5*f+4],  joints_position[5*f+2])+Distance_2XYZRGB( joints_position[5*f+5],  joints_position[5*f+2]));
            w[f].z = (temp_w[0].z*Distance_2XYZRGB( joints_position[5*f+3],  joints_position[5*f+2]) +
                      temp_w[1].z*Distance_2XYZRGB( joints_position[5*f+4],  joints_position[5*f+2])+
                      temp_w[2].z*Distance_2XYZRGB( joints_position[5*f+5],  joints_position[5*f+2]))/
                    (Distance_2XYZRGB( joints_position[5*f+3],  joints_position[5*f+2])+Distance_2XYZRGB( joints_position[5*f+4],  joints_position[5*f+2])+Distance_2XYZRGB( joints_position[5*f+5],  joints_position[5*f+2]));
        }
        else if (number_of_joints == 2){
            w[f].x = (temp_w[0].x*Distance_2XYZRGB( joints_position[5*f+3],  joints_position[5*f+2]) +
                      temp_w[1].x*Distance_2XYZRGB( joints_position[5*f+4],  joints_position[5*f+2]))/
                    (Distance_2XYZRGB( joints_position[5*f+3],  joints_position[5*f+2])+Distance_2XYZRGB( joints_position[5*f+4],  joints_position[5*f+2]));
            w[f].y = (temp_w[0].y*Distance_2XYZRGB( joints_position[5*f+3],  joints_position[5*f+2]) +
                      temp_w[1].y*Distance_2XYZRGB( joints_position[5*f+4],  joints_position[5*f+2]))/
                    (Distance_2XYZRGB( joints_position[5*f+3],  joints_position[5*f+2])+Distance_2XYZRGB( joints_position[5*f+4],  joints_position[5*f+2]));
            w[f].z = (temp_w[0].z*Distance_2XYZRGB( joints_position[5*f+3],  joints_position[5*f+2]) +
                      temp_w[1].z*Distance_2XYZRGB( joints_position[5*f+4],  joints_position[5*f+2]))/
                    (Distance_2XYZRGB( joints_position[5*f+3],  joints_position[5*f+2])+Distance_2XYZRGB( joints_position[5*f+4],  joints_position[5*f+2]));
        }
        else{
            w[f] = temp_w[0];
        }
        std::cout << f << ": " << w[f] << std::endl;

        for(int b = 0; b<number_of_joints; b++){
            Point3d v;
            v.x = joints_position[5*f+3+b].x - joints_position[5*f+2].x;
            v.y = joints_position[5*f+3+b].y - joints_position[5*f+2].y;
            v.z = joints_position[5*f+3+b].z - joints_position[5*f+2].z;
            float dist = v.x*w[f].x + v.y*w[f].y + v.z*w[f].z;
            joints_position[5*f+3+b].x = joints_position[5*f+3+b].x - dist*w[f].x;
            joints_position[5*f+3+b].y = joints_position[5*f+3+b].y - dist*w[f].y;
            joints_position[5*f+3+b].z = joints_position[5*f+3+b].z - dist*w[f].z;

            float scale = bone_length[f][b+1]/1000.0/Distance_2XYZRGB(joints_position[5*f+3+b], joints_position[5*f+2+b]);
            joints_position[5*f+3+b].x = (joints_position[5*f+3+b].x - joints_position[5*f+2+b].x)*scale + joints_position[5*f+2+b].x;
            joints_position[5*f+3+b].y = (joints_position[5*f+3+b].y - joints_position[5*f+2+b].y)*scale + joints_position[5*f+2+b].y;
            joints_position[5*f+3+b].z = (joints_position[5*f+3+b].z - joints_position[5*f+2+b].z)*scale + joints_position[5*f+2+b].z;

            //            if(abs(Distance_2XYZRGB(joints_position[5*f+3+b], joints_position[5*f+2+b]) - bone_length[f][b+1]/1000.0)>0.001)
            //                std::cout << "No!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;


        }



    }
    ROS_INFO("Smoothed");
}

void articulate_HandModel_XYZRGB::trial(){
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out (new pcl::PointCloud<pcl::PointXYZ>);

    // Fill in the CloudIn data
    cloud_in->width    = 2;
    cloud_in->height   = 5;
    cloud_in->is_dense = false;
    cloud_in->points.resize (cloud_in->width * cloud_in->height);
    for (size_t i = 0; i < cloud_in->points.size (); ++i)
    {
        cloud_in->points[i].x = 1024 * rand () / (RAND_MAX + 1.0f);
        cloud_in->points[i].y = 1024 * rand () / (RAND_MAX + 1.0f);
        cloud_in->points[i].z = 1024 * rand () / (RAND_MAX + 1.0f);
    }
    std::cout << "Saved " << cloud_in->points.size () << " data points to input:"
              << std::endl;
    for (size_t i = 0; i < cloud_in->points.size (); ++i) std::cout << "    " <<
                                                                       cloud_in->points[i].x << " " << cloud_in->points[i].y << " " <<
                                                                       cloud_in->points[i].z << std::endl;
    *cloud_out = *cloud_in;
    cloud_out->points.resize (cloud_in->width * cloud_in->height);
    std::cout << "size:" << cloud_out->points.size() << std::endl;
    for (size_t i = 0; i < cloud_in->points.size (); ++i)
        cloud_out->points[i].x = cloud_in->points[i].x + 0.7f;
    std::cout << "Transformed " << cloud_in->points.size () << " data points:"
              << std::endl;
    for (size_t i = 0; i < cloud_out->points.size (); ++i)
        std::cout << "    " << cloud_out->points[i].x << " " <<
                     cloud_out->points[i].y << " " << cloud_out->points[i].z << std::endl;
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputCloud(cloud_in);
    icp.setInputTarget(cloud_out);
    pcl::PointCloud<pcl::PointXYZ> Final;
    icp.align(Final);
    std::cout << "has converged:" << icp.hasConverged() << " score: " <<
                 icp.getFitnessScore() << std::endl;
    std::cout << icp.getFinalTransformation() << std::endl;

    return;
}



















