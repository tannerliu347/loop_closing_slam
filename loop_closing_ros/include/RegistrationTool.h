#pragma once
#include "LoopClosureParam.h"
#include "RobustMatcher.h"
#include "Matchdata.hpp"
#include <glog/logging.h>

struct RegistrationData2d
{
    cv::Mat img_cur;
    cv::Mat img_candidate;
    cv::Mat descriptors_cur;
    cv::Mat descriptors_candidate;
    vector<cv::DMatch> matches;
    vector<cv::KeyPoint> points_cur;
    vector<cv::KeyPoint> points_candidate;
    cv::Mat F;
    int inlierCount = 0;
    

};
struct RegistrationData3d
{
    vector<cv::Point3f> currentPoints;
    vector<cv::Point3f> oldPoints;
    RelativePose rp;
};
class RegistrationTool
{
    public:
        RegistrationTool(){};
        bool Registration2d(RegistrationData2d& data);
        bool Registration3d(RegistrationData3d& data){
            LOG(FATAL) << "3dRegistration not implemented yet";
            exit(1);
        }
    
};