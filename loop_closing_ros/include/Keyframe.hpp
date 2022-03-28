#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <eigen3/Eigen/Core>
#include <icecream.hpp>
#include <string>
#include <DBoW3/DBoW3.h>
#include "opencv2/calib3d/calib3d.hpp"
#include <iostream>
using namespace std;

class Keyframe {
  public:
    cv::Mat              img;
    cv::Mat              depth;
    int16_t              frameID;
    int16_t              featureNum;
    int16_t              globalKeyframeID;
    vector<cv::KeyPoint> keypoints;
    vector<cv::KeyPoint> good_keypoints;
    cv::Mat              descriptors;
    vector<int32_t>      globalIDs;
    vector<cv::Point3f>  point_3d;

    // Keyframe(int16_t f_id, const cv::Mat kf, const cv::Mat dpt, const vector<cv::KeyPoint> &key_points, const cv::Mat dess)
    //     : frameID(f_id)
    //     , img(kf)
    //     , depth(dpt)
    //     , descriptors(dess)
    //     , featureNum(dess.rows)
    //     , keypoints(key_points) {
    // }
    Keyframe(int16_t f_id, const cv::Mat kf,  const cv::Mat d,const vector<cv::KeyPoint> &key_points, const vector<cv::KeyPoint> & good_key,const cv::Mat dess,const int gloablKeyframe)
        : frameID(f_id)
        , img(kf)
        , depth(d)
        , descriptors(dess)
        , featureNum(dess.rows)
        , keypoints(key_points)
        , good_keypoints(good_key)
        , globalKeyframeID(gloablKeyframe){
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
};