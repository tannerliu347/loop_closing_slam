#pragma once
//#include "LoopClosingTool.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <memory>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <eigen3/Eigen/Core>
#include <icecream.hpp>
#include <string>
#include <DBoW3/DBoW3.h>
#include <fbow/fbow.h>
#include <fbow/vocabulary_creator.h>
#include "opencv2/calib3d/calib3d.hpp"
#include "Keyframe.hpp"
#include <unordered_map>
#include "Matchdata.hpp"
#include <queue>
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
    int ransacIterations;
    double RansacThresh2d;
    int top_match;
    int framegap;
    bool create_databasefile;
    parameters(double fx,double fy,double cx,double cy,double u,double v):FX(fx),
                                                                                FY(fy),
                                                                                CX(cx),
                                                                                CY(cy){};
    parameters(){
        FX = 383.141;
        FY = 382.908;
        CX = 318.244;
        CY = 246.718;
        RansacThresh2d = 50;
        top_match = 10;
        PIXEL_TO_METER_SCALEFACTOR = 0.001;
        ransacReprojectionError = 8;
        ransacIterations = 100;
        framegap = 5;
        create_databasefile = false;
    };

};
class LoopClosingTool{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    LoopClosingTool(fbow::Vocabulary* pDB);
   // bool detect_loop_test(const cv::Mat& img);
    bool detect_loop(Matchdata& point_match);
    bool find_connection(Keyframe& frame,int& candidate_id,Matchdata& point_match);
    // remove wrong pair with ransac
    int ransac_featureMatching(Keyframe& current,Keyframe& candidate);
    //create feature
    void create_feature();
    void create_feature(std::vector<cv::KeyPoint> Keypoints);
    void assignNewFrame(const cv::Mat &img,const cv::Mat &depth,int gloablKeyframeId,std::vector<int> globalID);
    void generateKeyframe();
    void get2DfeaturePosition(vector<cv::Point2f> &point_2d, const vector<cv::KeyPoint> &good_kp2);
    void get3DfeaturePosition(vector<cv::Point3f> &point_3d, const cv::Mat &dpt1, const vector<cv::KeyPoint> &good_kp1);
    void set2DfeaturePosition(vector<cv::Point2f> &point_2d){
        this->point_2d.clear();
        this->point_2d = point_2d;
    }
    void set3DfeaturePosition(vector<cv::Point3f> &point_3d){
        this->point_3d.clear();
        this->point_3d = point_3d;
    }
    void eliminateOutliersPnP(Keyframe& current,Keyframe& candidate);
    //update this
    void create_camera_p(){
        parameter = parameters();
    }

    void assignRansacGuess(const Eigen::Matrix3f &rot, const Eigen::Vector3f &pos);
    Matchdata genearteNewGlobalId(Keyframe& candidate,vector<cv::DMatch>& returned_matches);
private:
    fbow::Vocabulary* pDB_;
    unsigned int frameGap_; // We consider frames within this range as too close
    float minScoreAccept_; // Disregard ones lower than this
    Eigen::MatrixXd* loopClosureMatch_;
    std::vector<Keyframe> keyframes_; //store keyframe class
    std::vector<cv::Mat> histKFs_; // store image alone, may be delete

    //current matches and feature point
    vector<cv::DMatch> good_matches;
        
    //for ransac outlier elimination
    cv::Mat camera_mat, distort;
    vector<cv::DMatch> ransac_matches;
    unordered_map<int16_t, int32_t>ransac_matches_id_map;
    vector<cv::Point2f> point_2d; //2d point of current 
    vector<cv::Point3f> point_3d;

    // ransac
    cv::Mat ransacRGuess, ransacTGuess;

    int featureType_;
    int featureCount_;
    int currentGlobalKeyframeId;
    cv::Mat currentImage;
    cv::Mat currentDepth;
    std::vector<int> current_globalIDs;
    std::vector<cv::KeyPoint> currentKeypoints;
    std::vector<cv::KeyPoint> goodKeypoints;
    std::vector<cv::KeyPoint> good_lastKeypoints;
    std::vector<cv::Mat> descriptors;
    cv::Mat currentDescriptors;
    parameters parameter;
    int id = 0;
    int lastLoopClosure_;
   //std::vector<KeyFrame> histKFs_
};
