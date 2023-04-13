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
#include <string>
#include <fbow/fbow.h>
#include <fbow/vocabulary_creator.h>
#include "opencv2/calib3d/calib3d.hpp"
#include "Keyframe.hpp"
#include <unordered_map>
#include "Matchdata.hpp"
#include <queue>
#include <sophus/se3.hpp>
#include <memory>
#include "camera.h"
#include "LoopClosureParam.h"
#include "landmark_manager.h"
#include "RobustMatcher.h"
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <glog/logging.h>

#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearEquality.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/linear/Preconditioner.h>
#include <gtsam/linear/PCGSolver.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam_unstable/nonlinear/BatchFixedLagSmoother.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>
#include <gtsam/slam/SmartProjectionPoseFactor.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/SimpleCamera.h>
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
        PIXEL_TO_METER_SCALEFACTOR = 0.001;
        create_databasefile = false;
    };

};
class LoopClosingTool{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    LoopClosingTool(fbow::Vocabulary* pDB,shared_ptr<Camera> camera,shared_ptr<LoopClosureParam> config);
   // bool detect_loop_test(const cv::Mat& img);
    bool detect_loop(vector<Matchdata>& point_matches,gtsam::Pose3 state,vector<int>& connectedFrames);
    bool find_connection(Keyframes& frame,int& candidate_id,Matchdata& point_match);
    // remove wrong pair with ransac
    void eliminateOutliersFundamental(Keyframes& current,Keyframes& candidate);
    int ransac_featureMatching(Keyframes& current,Keyframes& candidate);
    void pnpCorrespondence(Keyframes& current,Keyframes& candidate);
    bool visualizePointMatch(int landmarkID,cv::Point2f point,cv::Point2f projectedLocation);
    void visualizeProjectionFrame(set<int> currentFrameId, set<int> candidateFrameId);
    void visualizeFrame(set<int> frames);
    //create feature
    void create_feature();
    void create_feature(std::vector<cv::KeyPoint>& Keypoints,cv::Mat descriptors);
    void assignNewFrame(const cv::Mat &img,const cv::Mat &depth,int gloablKeyframeId,std::vector<int> globalID);
    void generateKeyframe(vector<int>& connectedFrames);
    void get2DfeaturePosition(vector<cv::Point2f> &point_2d, const vector<cv::KeyPoint> &good_kp2);
    void get3DfeaturePosition(vector<cv::Point3f> &point_3d, const cv::Mat &dpt1, const vector<cv::KeyPoint> &good_kp1);
    void getInviewPoint(set<int>& inViewLandmark,set<int>& visited,int level,int startFrame);
    void set2DfeaturePosition(vector<cv::Point2f> &point_2d){
        this->point_2d.clear();
        this->point_2d = point_2d;
    }
    void set3DfeaturePosition(vector<cv::Point3f> &point_3d){
        this->point_3d.clear();
        this->point_3d = point_3d;
    }
    void eliminateOutliersPnP(Keyframes& current,Keyframes& candidate, RelativePose& pose);
    //update this
    void create_camera_p(){
        parameter = parameters();
    }

    void assignRansacGuess(const Eigen::Matrix3f &rot, const Eigen::Vector3f &pos);
    Matchdata genearteNewGlobalId(Keyframes& current, Keyframes& candidate,vector<cv::DMatch>& returned_matches,RelativePose& pose);

    // //set config
    // void setFeatureType(int featureType){
    //     featureType_ = featureType;
    // }

    // void setRansacP(double ransacReprojectionError, int ransacIterations){
    //     ransacReprojectionError_ = ransacReprojectionError;
    //     ransacIterations_ = ransacReprojectionError;
    // }
    // void setInlier_(int inlier){
    //     inlier_ = inlier;
    // }
    // void setTop_match(int top_match){
    //     top_match_ = top_match;
    // }
    // void setFrameskip(int skip_frame,int first_candidate,int near_frame){
    //     skip_frame_ = skip_frame;
    //     first_candidate_ = first_candidate;
    //     near_frame_ = near_frame;
    // }
    // void setMinScoreAccept(double minScoreAccept){
    //     minScoreAccept_ = minScoreAccept;
    // }
    void setVocabularyfile(string path){
        pDB_->readFromFile(path);
    }
    shared_ptr<LandmarkManagers> landmark_manager;
    unordered_map<int,int> processedID;
    vector<pair<shared_ptr<Landmarks>,int>> loopClosurePoint;
    vector<int> connections;
    shared_ptr<RobustMatcher> robust_matcher; 
private:
    fbow::Vocabulary* pDB_;
    float minScoreAccept_; // Disregard ones lower than this
    Eigen::MatrixXd* loopClosureMatch_;
    std::unordered_map<int,Keyframes> keyframes_; //store keyframe class
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
    // //config
    // int featureType_;
    // int featureCount_;
    // double ransacReprojectionError_;
    // int ransacIterations_;
    // double minimum_score;
    // int inlier_; // Minimum number of inlier
    // int top_match_; // Top N frame from fbow database to check for potential loop closure candidate
    // int first_candidate_; // first frame to start check for loop closure candidate
    // int skip_frame_; //number of frame to skip after find a loop closure candidate
    // int near_frame_; // number of frame to ignore when featching loop closure matching result
    //data
    int currentGlobalKeyframeId;
    cv::Mat currentImage;
    cv::Mat currentDepth;
    std::vector<int> current_globalIDs;
    std::vector<cv::KeyPoint> currentKeypoints;
    std::vector<cv::KeyPoint> goodKeypoints;
    std::vector<cv::KeyPoint> good_lastKeypoints;
    std::vector<cv::Point3f> good_lastpoint3d;
    std::vector<cv::Mat> descriptors;
    cv::Mat additionalDescriptors;
    cv::Mat currentDescriptors;
    parameters parameter;
    int id = 0;
    int lastLoopClosure_;
    int max_loopClosureId;

    //camera 
    shared_ptr<Camera> camera_;

    shared_ptr<LoopClosureParam> config_;

    std::unordered_map<int, gtsam::Pose3> states;
    unordered_map<int,int>  curKey_globalId_map;
   //std::vector<KeyFrame> histKFs_
};
