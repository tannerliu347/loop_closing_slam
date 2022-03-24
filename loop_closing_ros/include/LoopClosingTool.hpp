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
#include "Keyframe.hpp"
// class keyframe{

// };
struct parameters{
    //TODO: add all parameter, move this to a seprate header file
    double FX;
    double FY;
    double CX;
    double CY;
    double PIXEL_TO_METER_SCALEFACTOR;
    double ransacReprojectionError;
    double ransacIterations;
    double RansacThresh2d;
    int top_match;
    parameters(double fx,double fy,double cx,double cy,double u,double v):FX(fx),
                                                                                FY(fy),
                                                                                CX(cx),
                                                                                CY(cy){};
    parameters(){
        FX = 617.552978515625;
        FY = 618.0050659179688;
        CX = 323.6305236816406;
        CY = 242.8762664794922;
        RansacThresh2d = 20;
        top_match = 7;
        PIXEL_TO_METER_SCALEFACTOR = 0.001;
    };

};
class LoopClosingTool{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    LoopClosingTool(DBoW3::Database* pDB);
   // bool detect_loop_test(const cv::Mat& img);
    bool detect_loop(vector<int>& matchingindex);
    // remove wrong pair with ransac
    int ransac_featureMatching(Keyframe& current,Keyframe& candidate);
    //create feature
    void create_feature();
    void create_feature(cv::Mat &feature,std::vector<cv::KeyPoint> Keypoints);
    void assignNewFrame(const cv::Mat &img,const cv::Mat &depth,int gloablKeyframeId,std::vector<int> globalID);
    void generateKeyframe();
    void get2DfeaturePosition(vector<cv::Point2f> &point_2d, const vector<cv::KeyPoint> &good_kp2);
    void get3DfeaturePosition(vector<cv::Point3f> &point_3d, const cv::Mat &dpt1, const vector<cv::KeyPoint> &good_kp1);
    void set2DfeaturePosition(vector<cv::Point2f> &point_2d){
        this->point_2d = point_2d;
    }
    void set3DfeaturePosition(vector<cv::Point3f> &point_3d){
        this->point_3d = point_3d;
    }
    //update this
    void create_camera_p(){
        parameter = parameters();
    }
private:
    DBoW3::Database* pDB_;
    unsigned int frameGap_; // We consider frames within this range as too close
    float minScoreAccept_; // Disregard ones lower than this
    Eigen::MatrixXd* loopClosureMatch_;
    std::vector<Keyframe> keyframes_; //store keyframe class
    std::vector<cv::Mat> histKFs_; // store image alone, may be delete

    //current matches and feature point
    vector<cv::DMatch> good_matches;
    vector<cv::Point2f> point_2d; //2d point of current 
    vector<cv::Point3f> point_3d;
    int featureType_;
    int featureCount_;
    int currentGlobalKeyframeId;
    cv::Mat currentImage;
    cv::Mat currentDepth;
    std::vector<int> current_globalIDs;
    std::vector<cv::KeyPoint> currentKeypoints;
    cv::Mat currentDescriptors;
    parameters parameter;
    

   //std::vector<KeyFrame> histKFs_
};
