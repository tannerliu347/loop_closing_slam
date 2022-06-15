#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <eigen3/Eigen/Core>
#include <icecream.hpp>
#include <string>
#include "opencv2/calib3d/calib3d.hpp"
#include <iostream>
class Matchdata{
public:
    Matchdata(int curId,int oldId, vector<int> point_current,vector<int> point_old,vector<cv::KeyPoint> newmeasurement):curId_(curId),
                                                                                    oldId_(oldId),
                                                                                    point_current_(point_current),
                                                                                    point_old_(point_old),
                                                                                    newmeasurement_(newmeasurement){}
                                                                                    
    Matchdata(){};
    //TODO: make below code private, and set up interaface function
    int curId_;
    int oldId_;
    //vector<cv::DMatch> ransac_matches;
    vector<int> point_current_;
    vector<int> point_old_;
    vector<cv::KeyPoint> newmeasurement_;
};