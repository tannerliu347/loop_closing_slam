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
#include <sensor_msgs/CameraInfo.h>
#include <loop_closing_ros/LoopclosingConfig.h>
#include "LoopClosingManager.hpp"
#include "camera.h"
#include "config.h"
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
#include <frontend/FeatureUpdate.h>
//loop closing entry
LoopClosingManager loop_manager;
shared_ptr<Camera>              camera;
shared_ptr<Config>              config;
void filterCallback(const inekf_msgs::StateConstPtr &stateMsg,const frontend::Keyframe::ConstPtr& Framemsg) {
    ROS_INFO("I heard: [%d],  [%d]", loop_manager.frameCount,Framemsg->frameID);
    loop_manager.frameCount++;
    loop_manager.runLoopClosure(Framemsg,stateMsg);
};


void get_sensor_poses() {
    tf::TransformListener listener;
    tf::StampedTransform  transform;
    Eigen::Matrix3d       tmpMatrix;
    Eigen::Vector3d       tmpVector;
    try {
        listener.waitForTransform(config->cameraFrame, config->bodyFrame, ros::Time(0), ros::Duration(10.0));
        listener.lookupTransform(config->cameraFrame, config->bodyFrame, ros::Time(0), transform);
        
        config->cameraPoseInitialized = true;
        Eigen::Matrix3d tmpRot;
        Eigen::Vector3d tmpPos;
        tf::matrixTFToEigen(transform.getBasis(), tmpRot);
        tf::vectorTFToEigen(transform.getOrigin(), tmpPos);
        config->cameraPoseRotation    = tmpRot.cast<float>();
        config->cameraPoseTranslation = tmpPos.cast<float>();
        camera->updateExtrinsic(Sophus::SE3f(config->cameraPoseRotation,config->cameraPoseTranslation));
    } catch (tf::TransformException &ex) {
        ROS_ERROR("%s", ex.what());
    }
}

void dynamic_reconfig_callback(loop_closing_ros::LoopclosingConfig &configs, uint32_t level) {
    loop_manager.loopDetector->setVocabularyfile(configs.vocab_path);
    loop_manager.loopDetector->setFeatureType(configs.feature_type);
    loop_manager.loopDetector->setRansacP(configs.ransac_reprojection_error,configs.ransac_iterations);
    loop_manager.loopDetector->setInlier_(configs.inlier);
    loop_manager.loopDetector->setTop_match(configs.top_match);
    loop_manager.loopDetector->setMinScoreAccept(configs.min_score);
    loop_manager.loopDetector->setFrameskip(configs.skip_frame,configs.first_candidate,configs.near_frame);
}
void handle_camera_info(sensor_msgs::CameraInfo::Ptr msg) {
    // config->cameraInfoInitialized = true;
    // config->cameraMatrix          = (cv::Mat1d(3, 3) << msg->K[0], 0, msg->K[2], 0, msg->K[4], msg->K[5], 0, 0, 1);
    config->imageWidth            = msg->width;
    config->imageHeight           = msg->height;
    camera->updateParameter( msg->K[0],msg->K[4],msg->K[2], msg->K[5]);
    loop_manager.cameraIntialized = true;
}
void handle_landmark_update(frontend::FeatureUpdate::Ptr msg){
    loop_manager.loopDetector->landmark_manager->updateLandmark(msg->globalIDs,msg->points);
    ROS_INFO_STREAM("Receive point update_loopclosing");
}
int main(int argc, char **argv)
{
  if (ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Debug))
  {
        ros::console::notifyLoggerLevelsChanged();
  }
  camera.reset(new Camera(1,1,1,1));
  config.reset(new Config);
  ros::init(argc, argv, "loop_closing");
  ros::NodeHandle nh;
  string vocab_path;
  fbow::Vocabulary voc;
  //voc.readFromFile("/root/ws/curly_slam/catkin_ws/sift.fbow");
  //voc.readFromFile(vocab_path);
  LoopClosingTool lct(&voc,camera,config);
  //set up loop closing
  loop_manager.setCore(&nh,&lct);

  dynamic_reconfigure::Server<loop_closing_ros::LoopclosingConfig> server;
  server.setCallback(boost::bind(&dynamic_reconfig_callback, _1, _2));


  //get parameters
  string keyframe_topic;
  string state_topic;
  string camera_info;
  nh.param<string>("depth_topic",  keyframe_topic, "/backend/keyframe_out");
  nh.param<string>("state_topic", state_topic,  "/backend/state_out");
  nh.param<string>("camera_info_topic", camera_info,  "/camera/color/camera_info");
  //message filters
  message_filters::Subscriber<frontend::Keyframe> keyframe_sub_(nh, keyframe_topic, 5000);
  message_filters::Subscriber<inekf_msgs::State> state_sub_(nh, state_topic, 5000);
  typedef message_filters::sync_policies::ApproximateTime<inekf_msgs::State, frontend::Keyframe> SyncPolicy;
  message_filters::Synchronizer<SyncPolicy> sync(SyncPolicy(5000), state_sub_,  keyframe_sub_);
  sync.registerCallback(boost::bind(&filterCallback, _1, _2));
//message_filters::Subscriber<sensor_msgs::msg::Image> disparity_sub_;
  // set camera info
  ros::Subscriber subCameraInfo = nh.subscribe(camera_info, 10, &handle_camera_info);
  //update optimized Point
  ros::Subscriber subLandmark = nh.subscribe("/backend/landmarksId_out", 10, &handle_landmark_update);
  // load poses
  get_sensor_poses();
  ros::spin();

  return 0;
}