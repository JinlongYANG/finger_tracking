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
    Model_joints[1].at<float>(2,0) = 0.01;
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
    parameters[7] = 30;
    parameters[8] = 45;
    parameters[9] = 20;
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
    parameters[17] = 0;
    //18: horizontal angle between ring finger proximal and palm;
    //19: vertical angle between ring finger proximal and palm;
    //20: angle between ring finger proximal and intermediate;
    //21: angle between ring finger intermediate and distal;
    parameters[18] = -10;
    parameters[19] = 0;
    parameters[20] = 0;
    parameters[21] = 0;
    //22: horizontal angle between pinky proximal and palm;
    //23: vertical angle between pinky proximal and palm;
    //24: angle between pinky proximal and intermediate;
    //25: angle between pinky intermediate and distal;
    parameters[22] = -25;
    parameters[23] = 0;
    parameters[24] = 0;
    parameters[25] = 0;
}

void articulate_HandModel_XYZRGB::get_parameters(){

    //1. find hand roll pitch yaw(parameter 3, 4, 5)
    //1.1 determin the translation matrix of palm;
    Mat palm_model, palm;
    palm_model = Mat::zeros(3, 9, CV_32FC1);
    palm = Mat::zeros(3, 9, CV_32FC1);

    Model_joints[1].copyTo(palm_model.col(0));
    Model_joints[6].copyTo(palm_model.col(1));
    Model_joints[7].copyTo(palm_model.col(2));
    Model_joints[11].copyTo(palm_model.col(3));
    Model_joints[12].copyTo(palm_model.col(4));
    Model_joints[16].copyTo(palm_model.col(5));
    Model_joints[17].copyTo(palm_model.col(6));
    Model_joints[21].copyTo(palm_model.col(7));
    Model_joints[22].copyTo(palm_model.col(8));

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

    Mat R,t;
    poseEstimate::poseestimate::compute(palm,palm_model,R,t);

    //1.2 get the angles:
    cv::Mat mtxR, mtxQ;
    cv::Vec3d angles;
    angles = cv::RQDecomp3x3(R, mtxR, mtxQ);

    parameters[3] = angles[0];
    parameters[4] = angles[1];
    parameters[5] = angles[2];

    //2. find angles of every joints of index finger:



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
    joints_for_calc[8].at<float>(1,0) = bone_length[1][1]/1000.0;
    joints_for_calc[9].at<float>(1,0) = bone_length[1][2]/1000.0;
    joints_for_calc[10].at<float>(1,0) = bone_length[1][3]/1000.0;

    Mat R[3];
    R[0] = R_z(parameters[10])*R_x(parameters[11]);
    R[1] = R_x(parameters[12]);
    R[2] = R_x(parameters[13]);

    joints_for_calc[10] = R[0]*(R[1]*(R[2]*joints_for_calc[10]+joints_for_calc[9])+joints_for_calc[8])+joints_for_calc[7];
    joints_for_calc[9] = R[0]*(R[1]*joints_for_calc[9]+joints_for_calc[8])+joints_for_calc[7];
    joints_for_calc[8] = R[0]*joints_for_calc[8]+joints_for_calc[7];

    //2.2 middel to pinky(extrinsic):
    for ( int i = 0; i < 3; ++i){
        joints_for_calc[i*5+13].at<float>(1,0) = bone_length[2+i][1]/1000;
        joints_for_calc[i*5+14].at<float>(1,0) = bone_length[2+i][2]/1000;
        joints_for_calc[i*5+15].at<float>(1,0) = bone_length[2+i][3]/1000;

        R[0] = R_z(parameters[i*4+14])*R_x(parameters[i*4+15]);
        R[1] = R_x(parameters[i*4+16]);
        R[2] = R_x(parameters[i*4+17]);

        joints_for_calc[i*5+15] = R[0]*(R[1]*(R[2]*joints_for_calc[i*5+15]+joints_for_calc[i*5+14])+joints_for_calc[i*5+13])+joints_for_calc[i*5+12];
        joints_for_calc[i*5+14] = R[0]*(R[1]*joints_for_calc[i*5+14]+joints_for_calc[i*5+13])+joints_for_calc[i*5+12];
        joints_for_calc[i*5+13] = R[0]*joints_for_calc[i*5+13]+joints_for_calc[i*5+12];

    }

    //2.3 thumb(extrinsic)
    joints_for_calc[2].at<float>(1,0) = bone_length[0][0]/1000.0;
    joints_for_calc[3].at<float>(1,0) = bone_length[0][1]/1000.0;
    joints_for_calc[4].at<float>(1,0) = bone_length[0][2]/1000.0;

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
    }


}

void articulate_HandModel_XYZRGB::set_joints_positions(){

}


void articulate_HandModel_XYZRGB::expectation(std::vector<pcl::PointXYZRGB> & hand_pcl){

}

void articulate_HandModel_XYZRGB::maximization(std::vector<pcl::PointXYZRGB> & hand_pcl){

}
