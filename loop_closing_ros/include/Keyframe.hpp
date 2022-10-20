#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <eigen3/Eigen/Core>
#include <string>
#include "opencv2/calib3d/calib3d.hpp"
#include <iostream>
#include <set>
using namespace std;

class Keyframe {
  public:
    cv::Mat              img;
    cv::Mat              depth;
    int                  featureNum;
    int                  globalKeyframeID;
    vector<cv::KeyPoint> keypoints;
    cv::Mat              descriptors;
    vector<int>          globalIDs;
    vector<cv::Point3f>  point_3d;
    std::set<int>             connectedFrame;

    // Keyframe(int16_t f_id, const cv::Mat kf, const cv::Mat dpt, const vector<cv::KeyPoint> &key_points, const cv::Mat dess)
    //     : frameID(f_id)
    //     , img(kf)
    //     , depth(dpt)
    //     , descriptors(dess)
    //     , featureNum(dess.rows)
    //     , keypoints(key_points) {
    // }
    Keyframe(){
        globalKeyframeID = -1;
    }
    Keyframe(int32_t f_id, const cv::Mat kf,  const cv::Mat d,const vector<cv::KeyPoint> &key_points, const cv::Mat dess)
        : globalKeyframeID(f_id)
        , img(kf)
        , depth(d)
        , featureNum(dess.rows)
        , keypoints(key_points){
        if (dess.empty()){
            cout << "failure empty " << endl;
        }
        descriptors = dess;
    }

    void insertGlobalID(int16_t g_id) {
        globalIDs.push_back(g_id);
    }
    void insertGlobalID(vector<int32_t>& g_ids) {
        globalIDs = g_ids;
    }
    void insertPoint3D(vector<cv::Point3f>& p3d){
        point_3d = p3d;
    }
    void updateDescriptors(cv::Mat descriptors){
        this->descriptors =descriptors;
    }
};