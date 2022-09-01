#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <eigen3/Eigen/Core>
#include <string>
#include "opencv2/calib3d/calib3d.hpp"
#include <iostream>
#include <unordered_map>
#include  <memory> 
#include "ros/ros.h"
using namespace std;
class Landmark{
public: 
    int globalId;
    cv::Mat descriptor;
    Landmark(int globalId):globalId(globalId),inView(false){
        
    }
    cv::Point3f getGlobalPos(){
        return globalPos;
    }
    void updatePoint(cv::Point3f newData,bool inView){
        globalPos = newData;
        this->inView = inView;
    }
    bool getinView(){
        return inView;
    }
    
private:
    bool inView;
    cv::Point3f globalPos;
    
};
class Landmarks{
public: 
    cv::Point3f operator[] (int globalID){
        if (points3d.find(globalID) != points3d.end()){
            return points3d[globalID]->getGlobalPos();
        }
    }
    cv::Mat getDescriptor(int globalID){
        return points3d[globalID] ->descriptor;
    }
    void update(const vector<int32_t>& globalIDs, const vector<cv::Point3f>&  newPoint ,const vector<bool>&  inViews ){
        //points3d[globalIDs] = newPoint[globalIDs];
        for (int i = 0; i < globalIDs.size(); i ++){
            int globalID = globalIDs[i];
            if (points3d.find(globalID) == points3d.end()){
                points3d[globalID].reset(new Landmark(globalID));
            }else{
                points3d[globalID]->updatePoint(newPoint[i],inViews[i]);
            }
            
        }
    }
    void updateDescriptor(const vector<int32_t>& globalIDs, cv::Mat& discriptrors){
          ROS_DEBUG_STREAM("globalID size" <<   globalIDs.size() << "descriptor size " << discriptrors.rows);
        int i = 0;
        for(auto globalID:globalIDs){
            points3d[globalID]->descriptor = discriptrors.row(i);
            i++;
        }
    };
    vector<cv::Point3f> get3dPoint(const vector<int32_t>& globalIDs){
        vector<cv::Point3f> pointsOut;
        for (auto globalID: globalIDs ){
            if (points3d.find(globalID) == points3d.end()){
                ROS_ERROR_STREAM("cannot find landmark" << globalID);
            }else{
                pointsOut.push_back(points3d[globalID]->getGlobalPos());
            }
            
        }
        return pointsOut;
    }
    vector<shared_ptr<Landmark>> getInView3dPoint(){
        vector<shared_ptr<Landmark>> inViewLandmarks;
        for (auto landmarks: points3d){
            if (landmarks.second->getinView()){
                inViewLandmarks.push_back(landmarks.second);
            }
        }
        return inViewLandmarks;
    }
    unordered_map<int,shared_ptr<Landmark>> points3d;

};
class Keyframe {
  public:
    cv::Mat              img;
    cv::Mat              depth;
    int32_t              featureNum;
    int32_t              globalKeyframeID;
    vector<cv::KeyPoint> keypoints;
    cv::Mat              descriptors;
    vector<int32_t>      globalIDs;
    shared_ptr<Landmarks> ldmarks;
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
    Keyframe(int32_t f_id, const cv::Mat kf,  const cv::Mat d,const vector<cv::KeyPoint> &key_points, const cv::Mat dess, shared_ptr<Landmarks> ldmarks)
        : globalKeyframeID(f_id)
        , img(kf)
        , depth(d)
        , featureNum(dess.rows)
        , keypoints(key_points)
        , ldmarks(ldmarks){
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
    vector<cv::Point3f> get3dPoint(){
        return ldmarks->get3dPoint(globalIDs);
    }
};