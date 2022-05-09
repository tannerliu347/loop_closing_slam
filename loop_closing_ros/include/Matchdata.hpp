#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <eigen3/Eigen/Core>
#include <icecream.hpp>
#include <string>
#include <DBoW3/DBoW3.h>
#include "opencv2/calib3d/calib3d.hpp"
#include <iostream>
class Matchdata{
public:
    Matchdata(){

    }
private:
    int curFrameIdx;
    int matchFrameIdx;
    vector<cv::DMatch> ransac_matches;
    vector<int> point_match_current;
    vector<int> point_match_old;
};