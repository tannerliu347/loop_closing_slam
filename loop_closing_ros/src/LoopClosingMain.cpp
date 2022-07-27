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
#include "LoopClosingManager.hpp"
//loop closing entry
LoopClosingManager loop_manager;
void filterCallback(const inekf_msgs::StateConstPtr &stateMsg,const frontend::Keyframe::ConstPtr& Framemsg) {
    ROS_INFO("I heard: [%d],  [%d]", loop_manager.frameCount,Framemsg->frameID);
    loop_manager.frameCount++;
    loop_manager.runLoopClosure(Framemsg,stateMsg);
};



void dynamic_reconfig_callback(loop_closing_ros::LoopclosingConfig &config, uint32_t level) {
    loop_manager.loopDetector->setVocabularyfile(config.vocab_path);
    loop_manager.loopDetector->setFeatureType(config.feature_type);
    loop_manager.loopDetector->setRansacP(config.ransac_reprojection_error,config.ransac_iterations);
    loop_manager.loopDetector->setInlier_(config.inlier);
    loop_manager.loopDetector->setTop_match(config.top_match);
    loop_manager.loopDetector->setMinScoreAccept(config.min_score);
    loop_manager.loopDetector->setFrameskip(config.skip_frame,config.first_candidate,config.near_frame);
}

int main(int argc, char **argv)
{
  if (ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Debug))
  {
        ros::console::notifyLoggerLevelsChanged();
  }
  ros::init(argc, argv, "loop_closing");
  ros::NodeHandle nh;
  string vocab_path;
  fbow::Vocabulary voc;
  //voc.readFromFile("/root/ws/curly_slam/catkin_ws/sift.fbow");
  //voc.readFromFile(vocab_path);
  LoopClosingTool lct(&voc);
  //set up loop closing
  loop_manager.setCore(&nh,&lct);

  dynamic_reconfigure::Server<loop_closing_ros::LoopclosingConfig> server;
  server.setCallback(boost::bind(&dynamic_reconfig_callback, _1, _2));


  //get parameters
  string keyframe_topic;
  string state_topic;
  nh.param<string>("depth_topic",  keyframe_topic, "/backend/keyframe_out");
  nh.param<string>("state_topic", state_topic,  "/backend/state_out");
  //message filters
  message_filters::Subscriber<frontend::Keyframe> keyframe_sub_(nh, keyframe_topic, 5000);
  message_filters::Subscriber<inekf_msgs::State> state_sub_(nh, state_topic, 5000);
  typedef message_filters::sync_policies::ApproximateTime<inekf_msgs::State, frontend::Keyframe> SyncPolicy;
  message_filters::Synchronizer<SyncPolicy> sync(SyncPolicy(5000), state_sub_,  keyframe_sub_);
  sync.registerCallback(boost::bind(&filterCallback, _1, _2));
//message_filters::Subscriber<sensor_msgs::msg::Image> disparity_sub_;
  ros::spin();

  return 0;
}