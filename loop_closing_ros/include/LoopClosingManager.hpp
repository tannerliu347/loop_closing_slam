#pragma once
#include "ros/ros.h"
#include "LoopClosingTool.hpp"
#include "std_msgs/String.h"
#include "frontend/Keyframe.h"
#include "frontend/Match.h"
#include "Matchdata.hpp"
#include "string"
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <inekf_msgs/State.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/Point.h>
#include <unordered_map>
#include <sophus/se3.hpp>
#include <dynamic_reconfigure/server.h>
#include <loop_closing_ros/LoopclosingConfig.h>
class LoopClosingManager{
public:
    LoopClosingManager(){markerId = 0;}
    
    //set up loop closure tool
    void setCore(ros::NodeHandle* nh, LoopClosingTool* ltr);
    void runLoopClosure(const frontend::Keyframe::ConstPtr& msg,const inekf_msgs::StateConstPtr &stateMsg);
    void publishMatch(Matchdata& point_match);
    void drawLine(int i);
    void drawLine(const vector<int>& matchingIndex);
    void drawPoint(const geometry_msgs::Pose pose);
    int frameCount;
    LoopClosingTool* loopDetector;
private: 
    ros::NodeHandle* nh;
    //ros::Subscriber keyframeSub_;
    ros::Publisher closingLine_pub;
    ros::Publisher keyframe_pub;
    ros::Publisher match_pub;
    ros::Publisher test_point_pub;
    std::vector<frontend::Keyframe> keyframes;
    std::unordered_map<int, inekf_msgs::State> states;
    inekf_msgs::State current_state;
    int markerId;
   
    //subscriber

};
