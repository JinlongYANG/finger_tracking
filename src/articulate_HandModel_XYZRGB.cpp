#include "finger_tracking/articulate_HandModel_XYZRGB.h"
#include <opencv2/calib3d/calib3d.hpp>
#include "finger_tracking/poseestimate.hpp"

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
    bone_length[1][0] = 79.4271;
    bone_length[1][1] = 48.0471;
    bone_length[1][2] = 28.7806;
    bone_length[1][3] = 21.517;
    //middle finger
    bone_length[2][0] = 75.5294;
    bone_length[2][1] = 53.4173;
    bone_length[2][2] = 33.1543;
    bone_length[2][3] = 23.2665;
    //ring finger
    bone_length[3][0] = 68.2215;
    bone_length[3][1] = 49.8076;
    bone_length[3][2] = 32.4014;
    bone_length[3][3] = 23.1557;
    //pinky finger
    bone_length[4][0] = 63.4492;
    bone_length[4][1] = 40.2519;
    bone_length[4][2] = 24.0526;
    bone_length[4][3] = 21.672;

    //3. Model joints position initialization
    for(int i = 0; i < 26; i++){
        Model_joints[i] = Mat::zeros(3,1,CV_32FC1);
    }
    //3.1. palm joints: 1,6,11,16,21,7,12,17,22
    //palm joints with reference to palm/hand coordinate:
    //palm.thumb
    Model_joints[1].at<float>(0,0) = -0.019;
    Model_joints[1].at<float>(1,0) = -0.054;
    Model_joints[1].at<float>(2,0) = 0.001;
    //palm.index
    Model_joints[6].at<float>(0,0) = -0.012;
    Model_joints[6].at<float>(1,0) = -0.049;
    Model_joints[6].at<float>(2,0) = -0.008;

    Model_joints[7].at<float>(0,0) = -0.027;
    Model_joints[7].at<float>(1,0) = 0.019;
    Model_joints[7].at<float>(2,0) = 0;
    //palm.middle
    Model_joints[11].at<float>(0,0) = 0;
    Model_joints[11].at<float>(1,0) = -0.050;
    Model_joints[11].at<float>(2,0) = -0.008;

    Model_joints[12].at<float>(0,0) = 0;
    Model_joints[12].at<float>(1,0) = 0.024;
    Model_joints[12].at<float>(2,0) = 0;
    //palm.ring
    Model_joints[16].at<float>(0,0) = 0.010;
    Model_joints[16].at<float>(1,0) = -0.049;
    Model_joints[16].at<float>(2,0) = -0.008;

    Model_joints[17].at<float>(0,0) = 0.018;
    Model_joints[17].at<float>(1,0) = 0.019;
    Model_joints[17].at<float>(2,0) = 0;
    //palm.pinky
    Model_joints[21].at<float>(0,0) = 0.020;
    Model_joints[21].at<float>(1,0) = -0.049;
    Model_joints[21].at<float>(2,0) = -0.008;

    Model_joints[22].at<float>(0,0) = 0.036;
    Model_joints[22].at<float>(1,0) = 0.015;
    Model_joints[22].at<float>(2,0) = 0;
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

    Model_joints[1].copyTo(palm_model.col(0));
    Model_joints[6].copyTo(palm_model.col(1));
    Model_joints[7].copyTo(palm_model.col(2));
    Model_joints[11].copyTo(palm_model.col(3));
    Model_joints[12].copyTo(palm_model.col(4));
    Model_joints[16].copyTo(palm_model.col(5));
    Model_joints[17].copyTo(palm_model.col(6));
    Model_joints[21].copyTo(palm_model.col(7));
    Model_joints[22].copyTo(palm_model.col(8));

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

    R = R_z(parameters[6]+10)*R_x(parameters[7])*R_y(50);

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
    R[0] = R_y(50);
    R[0] = R_z(10+parameters[6])*R_x(parameters[7])*R[0];
    R[1] = R_x(parameters[8]);
    R[2] = R_x(parameters[9]);

    //    cv::Mat mtxR, mtxQ;
    //    cv::Vec3d angles;
    //    angles = cv::RQDecomp3x3(R[0]*R_y(50).inv(), mtxR, mtxQ);
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
    for(int i = 0; i< 26; i++){
        joints_for_calc[i] = R_p_r_y * joints_for_calc[i];
    }



    //4. put calculation results into joints_position
    for(int i = 0; i< 26; i++){
        joints_position[i].x = joints_for_calc[i].at<float>(0,0);
        joints_position[i].y = joints_for_calc[i].at<float>(0,1);
        joints_position[i].z = joints_for_calc[i].at<float>(0,2);
        //std::cout<< i <<": "<<joints_position[i]<<std::endl;
    }



}

void articulate_HandModel_XYZRGB::set_joints_positions(){

}

void articulate_HandModel_XYZRGB::CP_palm_fitting1(Mat Hand_DepthMat,Mat LabelMat, int resolution){
    //1. ICP palm:
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
    for(int i = 0; i<8; i++){
        std::cout <<i<<": " << temp_row[i] <<" " << temp_col[i]<< " " << temp_dis[i]<<std::endl;
    }

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
    poseEstimate::poseestimate::compute(Oberservation,palm_model,R,t);

//    std::cout<<"Ober: "<< Oberservation << std::endl;
//    std::cout << "palm: " << palm << std::endl;

//    std::cout<<"R: "<< R << std::endl;
//    std::cout << "t: " << t << std::endl;
    //1.3.3 get the angles:
    cv::Mat mtxR, mtxQ;
    cv::Vec3d angles;
    angles = cv::RQDecomp3x3(R, mtxR, mtxQ);
    std::cout<<"angles: " << angles <<std::endl;

    parameters[3] = angles[0];
    parameters[4] = angles[1];
    parameters[5] = angles[2];


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
                count[0]+=length;
                direction[0].x += temp_direction.x;
                direction[0].y += temp_direction.y;
                direction[0].z += temp_direction.z;

                break;
            }

                //index finger proximal
            case 8:
            {
                temp_direction.z = (Hand_DepthMat.at<unsigned char>(row, col)-imageSize/2.0)*resolution/1000.0 - joints_position[7+bone].z;
                temp_direction.x = (col-imageSize/2.0)*resolution/1000.0 - joints_position[7+bone].x;
                temp_direction.y = (row-imageSize/2.0)*resolution/1000.0 - joints_position[7+bone].y;


                length = sqrt(temp_direction.x*temp_direction.x+temp_direction.y*temp_direction.y+temp_direction.z*temp_direction.z);
                                count[1]+=length;
                direction[1].x += temp_direction.x;
                direction[1].y += temp_direction.y;
                direction[1].z += temp_direction.z;

                break;
            }

                //middle finger proximal
            case 11:
            {
                temp_direction.z = (Hand_DepthMat.at<unsigned char>(row, col)-imageSize/2.0)*resolution/1000.0 - joints_position[12+bone].z;
                temp_direction.x = (col-imageSize/2.0)*resolution/1000.0 - joints_position[12+bone].x;
                temp_direction.y = (row-imageSize/2.0)*resolution/1000.0 - joints_position[12+bone].y;


                length = sqrt(temp_direction.x*temp_direction.x+temp_direction.y*temp_direction.y+temp_direction.z*temp_direction.z);
                count[2]+=length;
                direction[2].x += temp_direction.x;
                direction[2].y += temp_direction.y;
                direction[2].z += temp_direction.z;

                break;
            }

                //ring finger proximal
            case 14:
            {
                temp_direction.z = (Hand_DepthMat.at<unsigned char>(row, col)-imageSize/2.0)*resolution/1000.0 - joints_position[17+bone].z;
                temp_direction.x = (col-imageSize/2.0)*resolution/1000.0 - joints_position[17+bone].x;
                temp_direction.y = (row-imageSize/2.0)*resolution/1000.0 - joints_position[17+bone].y;


                length = sqrt(temp_direction.x*temp_direction.x+temp_direction.y*temp_direction.y+temp_direction.z*temp_direction.z);
                count[3]+=length;
                direction[3].x += temp_direction.x;
                direction[3].y += temp_direction.y;
                direction[3].z += temp_direction.z;

                break;
            }

                //little finger proximal
            case 17:
            {
                temp_direction.z = (Hand_DepthMat.at<unsigned char>(row, col)-imageSize/2.0)*resolution/1000.0 - joints_position[22+bone].z;
                temp_direction.x = (col-imageSize/2.0)*resolution/1000.0 - joints_position[22+bone].x;
                temp_direction.y = (row-imageSize/2.0)*resolution/1000.0 - joints_position[22+bone].y;


                length = sqrt(temp_direction.x*temp_direction.x+temp_direction.y*temp_direction.y+temp_direction.z*temp_direction.z);
                count[4]+=length;
                direction[4].x += temp_direction.x;
                direction[4].y += temp_direction.y;
                direction[4].z += temp_direction.z;

            }

            default:
            {
                ;
            }

            }
        }
    }

    for(int i = 0; i< 5; i++){
        direction[i].x = direction[i].x/count[i];
        direction[i].y = direction[i].y/count[i];
        direction[i].z = direction[i].z/count[i];
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




















