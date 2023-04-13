
#ifndef LANDMARK_MANAGER_H
#define LANDMARK_MANAGER_H

#include <Eigen/Eigen>
#include <unordered_map>
#include <unordered_set>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <sophus/se3.hpp>
#include "Keyframe.hpp"
#include "camera.h"
#include "LoopClosureParam.h"
using namespace std;

struct Landmarks {
    int                       landmarkId;
    size_t                    measurementSize = 0;
    int                       inViewCount     = 0;
    Eigen::Vector3f           pointGlobal; 
    vector<int>         observedFrameIds;
    map<int,cv::KeyPoint>      keypoints;  
    cv::Mat                   descriptor; 
    bool                      optimized;
    bool                      initiated;
    Landmarks(){
        optimized = false;
        initiated    = false;
     
   
    }
    void updateDescriptor(cv::Mat newDescriotor){
        // descriptor = newDescriotor;
        if (!initiated){
            descriptor = newDescriotor;
        }
        else{
            float newNorm = cv::norm(newDescriotor,cv::NORM_L2);
            float oldNorm = cv::norm(descriptor,cv::NORM_L2);
            if (newNorm < oldNorm){
                descriptor = newDescriotor;
            }
        }
    }
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct SearchResult {
    float        score;
    cv::KeyPoint keypoint; // for loop closing
    cv::Point2f  positionUV;
    float        depth;
    int          keypointIndex;
    SearchResult(float score_, cv::KeyPoint keypoint_, cv::Point2f positionUV_, float depth_, int keypointIndex_)
        : score(score_)
        , keypoint(keypoint_)
        , positionUV(positionUV_)
        , depth(depth_)
        , keypointIndex(keypointIndex_) {
    }
};
class LandmarkManagers {

  private:
    shared_ptr<LoopClosureParam> config;
    bool inView(int LandmarkID,Sophus::SE3f T_w_i,cv::Point2f& projectedLocation);
    vector<int> currentProcessingGlobalId;
  public:
    unordered_map<int, shared_ptr<Landmarks>> landmarks;
    shared_ptr<Camera>                     camera;
    void addKeyframe(Keyframes& keyframe);
    LandmarkManagers(shared_ptr<LoopClosureParam> config,shared_ptr<Camera> camera)
        : config(config)
        , camera(camera){
    }
    cv::Mat getDescriptors(vector<int>& globalIDs);
    void updateLandmark(vector<int>& globalIds,vector<vector<double>>& points);
    void plotTrackingStatistic();
    vector<shared_ptr<Landmarks>> getVisibleMapPoint(int currentFrameId,Sophus::SE3f T_w_i,unordered_map<int,int>& processed,vector<cv::Point2f>& ProjectedLocations);
    
};
#endif