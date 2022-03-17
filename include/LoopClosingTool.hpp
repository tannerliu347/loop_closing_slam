#pragma once
//#include "LoopClosingTool.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <eigen3/Eigen/Core>
#include <icecream.hpp>
#include <string>
#include <DBoW3/DBoW3.h>
#include "opencv2/calib3d/calib3d.hpp"
class LoopClosingTool{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    LoopClosingTool(DBoW3::Database* pDB,Eigen::MatrixXd* loopClosureMatch);
    bool detect_loop(const cv::Mat& img);
   // bool detect_loop_test(const cv::Mat& img);
    int ransac_featureMatching(const cv::Mat& curImg,const cv::Mat& candidateImg);
private:
    DBoW3::Database* pDB_;
    unsigned int frameGap_; // We consider frames within this range as too close
    float minScoreAccept_; // Disregard ones lower than this
    Eigen::MatrixXd* loopClosureMatch_;
    std::vector<cv::Mat> histKFs_;
   //std::vector<KeyFrame> histKFs_
};