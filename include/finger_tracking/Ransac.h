#include <opencv2/core/core.hpp>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;

void Ransac(vector<Point3d> Points3D, Point3d& center, int maximum_round, float range){

    Point3d chosen_center;
    srand((unsigned)time(NULL));
    float max_ratio = 0;
    int best_index = 0;
    /*****************  Ransac to find chosen center  *********************/
    for(int i = 0; i < maximum_round; i++){
        float random_index = rand() % Points3D.size();
//        cout<<"random idex:"<<random_index<<endl;

        chosen_center.x = Points3D[random_index].x;
        chosen_center.y = Points3D[random_index].y;
        chosen_center.z = Points3D[random_index].z;

        int inlier = 0;

        for(int j = 0; j < Points3D.size(); j++){
            if(abs(Points3D[j].x - chosen_center.x) < range &&
                    abs(Points3D[j].y - chosen_center.y) < range &&
                    abs(Points3D[j].z - chosen_center.z) < range){
                inlier++;
            }
        }
        float random_ration = inlier*1.0/Points3D.size();
        if(random_ration > max_ratio){
            max_ratio = random_ration;
            best_index = random_index;
        }

        if(max_ratio>0.9)
            break;
    }
    /***************   Calculate true center using all inliers **********/
    int inlier = 0;
    for(int j = 0; j < Points3D.size(); j++){
        if(abs(Points3D[j].x - Points3D[best_index].x) < range &&
                abs(Points3D[j].y - Points3D[best_index].y) < range &&
                abs(Points3D[j].z - Points3D[best_index].z) < range){
            center.x += Points3D[j].x;
            center.y += Points3D[j].y;
            center.z += Points3D[j].z;
            inlier++;
        }
    }
    center.x = center.x/inlier;
    center.y = center.y/inlier;
    center.z = center.z/inlier;

}

