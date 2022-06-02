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
class loop_closing_ros{
public:
    loop_closing_ros(){markerId = 0;}
    
    //set up loop closure tool
    void set_core(ros::NodeHandle* nh, LoopClosingTool* ltr){
        nh_ = nh;
        loopDetector_ = ltr;
        closingLine_pub = nh->advertise<visualization_msgs::Marker>("loop_closing_line", 10);
        //keyframe_pub = nh->advertise<frontend::Keyframe>("loop_closing/keyframe", 10);
        match_pub = nh->advertise<frontend::Match>("loop_closing/match", 10);
        frameCount = 0;
    }
    void run_loopClosure(const frontend::Keyframe::ConstPtr& msg,const inekf_msgs::StateConstPtr &stateMsg,int frameID){
        keyframes.push_back(*msg);
        current_state_ = (*stateMsg);
        states[frameID] = (*stateMsg);
        
        //current pose
        Eigen::Vector3f    positionVector(stateMsg->position.x, stateMsg->position.y, stateMsg->position.z);
        Eigen::Quaternionf poseOrientation(stateMsg->orientation.w, stateMsg->orientation.x, stateMsg->orientation.y, stateMsg->orientation.z);


        cv_bridge::CvImagePtr color_ptr;
        cv_bridge::CvImagePtr depth_ptr;
        //cv_bridge::CvImagePtr descriptor_ptr;
        sensor_msgs::ImageConstPtr colorImg( new sensor_msgs::Image( msg->color ) );
        sensor_msgs::ImageConstPtr depthImg( new sensor_msgs::Image( msg->depth ) );
        //sensor_msgs::ImageConstPtr descriptorImg( new sensor_msgs::Image( msg->descriptor) );
        try {
            color_ptr = cv_bridge::toCvCopy(colorImg, sensor_msgs::image_encodings::BGR8);
            depth_ptr = cv_bridge::toCvCopy(depthImg, sensor_msgs::image_encodings::TYPE_16UC1);
            //descriptor_ptr = cv_bridge::toCvCopy(descriptorImg,sensor_msgs::image_encodings::TYPE_16UC1);
        } catch (cv_bridge::Exception &e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
        cv::Mat color = color_ptr->image;
        cv::Mat depth = depth_ptr->image;
        //cv::Mat descriptor = descriptor_ptr->image;
        
        //extract globalid, and key points, 
        std::vector<int> globalId;
        std::vector<cv::Point2f> feature_2d;
        std::vector<cv::Point3f> feature_3d;
        std::vector<cv::KeyPoint> keypoints;
        for (int i = 0; i < msg-> features.size(); i ++){
            globalId.push_back(msg-> features[i].globalID);
            feature_2d.push_back(cv::Point2f(msg-> features[i].u,msg-> features[i].v));
            feature_3d.push_back(cv::Point3f(msg-> features[i].x,msg-> features[i].y,msg-> features[i].z));
            keypoints.push_back(cv::KeyPoint(cv::Point2f(msg-> features[i].u,msg-> features[i].v)
                                            ,msg-> features[i].size
                                            ,msg-> features[i].angle
                                            ,msg-> features[i].response
                                            ,msg-> features[i].octave
                                            ,msg-> features[i].class_id));
        }
        cv::imshow("dispaly",color);
        vector<int> matchingIndex;
        loopDetector_->assignNewFrame(color,depth,msg->frameID,globalId);
        loopDetector_->create_feature(keypoints);
        //loopDetector_->create_feature();
        // loopDetector_->set2DfeaturePosition(feature_2d);  
        // loopDetector_->set3DfeaturePosition(feature_3d);
        loopDetector_->assignRansacGuess(poseOrientation.toRotationMatrix(),positionVector);
        Matchdata point_match;
        bool loopdetected = loopDetector_->detect_loop(point_match);

        //cv::imwrite("image"+std::to_string(keyframes.size())+".jpg",color );
        if (loopdetected){
            //update globalId
            draw_line(point_match.oldId_);
            publish_match(point_match);
            //publish Matched data
        }
        // }else{
        //    // keyframe_pub.publish(*msg);
        // }

    }
    void publish_match(Matchdata& point_match){
        frontend::Match match_msg;
        match_msg.curId = point_match.curId_;
        match_msg.oldId = point_match.oldId_;
        match_msg.curPoint = point_match.point_current_;
        match_msg.oldPoint = point_match.point_old_;
        ROS_INFO_STREAM("curId " << match_msg.curPoint.size() << " oldId " << match_msg.oldPoint.size() << endl);
        for (int i = 0; i < match_msg.oldPoint.size(); i ++){
            ROS_INFO_STREAM(" curpointId_unpublished " << match_msg.curPoint[i] << " oldpointId_unpublished " << match_msg.oldPoint[i] );
        }
        vector<geometry_msgs::Point> measurement;
        for (auto keypoint:point_match.newmeasurement_){
            geometry_msgs::Point newPoint;
            newPoint.x = keypoint.pt.x;
            newPoint.y = keypoint.pt.y;
            measurement.push_back(newPoint);
        }
        match_msg.measurement = measurement;
        //TODO: publish pose 
        auto current_state = current_state_;
        auto old_state = states[match_msg.oldId];
        Eigen::Vector3f    position_cur(current_state.position.x, current_state.position.y, current_state.position.z);
        Eigen::Quaternionf poseOrientation_cur(current_state.orientation.w, current_state.orientation.x, current_state.orientation.y, current_state.orientation.z);
        Sophus::SE3f currentInekfPose(poseOrientation_cur,position_cur);

        Eigen::Vector3f    position_old(old_state.position.x, old_state.position.y, old_state.position.z);
        Eigen::Quaternionf poseOrientation_old(old_state.orientation.w, old_state.orientation.x, old_state.orientation.y, old_state.orientation.z);
        Sophus::SE3f oldInekfPose(poseOrientation_old,position_old);

        Sophus::SE3f relativePose = oldInekfPose.inverse() * currentInekfPose;
        auto t = relativePose.translation();
        auto q = relativePose.unit_quaternion().coeffs();

        geometry_msgs::Pose betweenPose;
        betweenPose.orientation.w = q.w();
        betweenPose.orientation.x = q.x();
        betweenPose.orientation.y = q.y();
        betweenPose.orientation.z = q.z();

        betweenPose.position.x = t.x();
        betweenPose.position.y = t.y();
        betweenPose.position.z = t.z();

        match_msg.betweenPose = betweenPose;
        match_pub.publish(match_msg);
    }
    void draw_line(int i){
        geometry_msgs::Point p_current;
        p_current.x = current_state_.position.x;
        p_current.y = current_state_.position.y;
        p_current.z = current_state_.position.z;
       
        visualization_msgs::Marker line_strip;
        line_strip.header.frame_id = "odom";
        line_strip.header.stamp = ros::Time::now();
        line_strip.ns =  "points_and_lines";
        line_strip.action = visualization_msgs::Marker::ADD;
        line_strip.pose.orientation.w = 1.0;
        line_strip.id = markerId++;
        line_strip.color.b = 1.0;
        line_strip.color.r = 1.0;
        line_strip.color.a = 1.0;
        line_strip.scale.x = 0.1;
        //matching point
        geometry_msgs::Point matching_point;
        matching_point.x = states[i].position.x;
        matching_point.y= states[i].position.y;
        matching_point.z = states[i].position.z;
        line_strip.points.push_back(p_current);
        line_strip.points.push_back(matching_point);
        closingLine_pub.publish(line_strip);
        

        
    }
    void draw_line(const vector<int>& matchingIndex){
        geometry_msgs::Point p_current;
        p_current.x = current_state_.position.x;
        p_current.y = current_state_.position.y;
        p_current.z = current_state_.position.z;
        for(int i = 0; i < matchingIndex.size();i++){
            visualization_msgs::Marker line_strip;
            line_strip.header.frame_id = "odom";
            line_strip.header.stamp = ros::Time::now();
            line_strip.ns =  "points_and_lines";
            line_strip.action = visualization_msgs::Marker::ADD;
            line_strip.pose.orientation.w = 1.0;
            line_strip.id = markerId++;
            line_strip.color.b = 1.0;
            line_strip.color.r = 1.0;
            line_strip.color.a = 1.0;
            line_strip.scale.x = 0.1;
            line_strip.scale.y = 0.1;
            line_strip.scale.z = 0.1;
            line_strip.type = visualization_msgs::Marker::LINE_STRIP;
            //matching point
            geometry_msgs::Point matching_point;
            matching_point.x = states[i].position.x;
            matching_point.y= states[i].position.y;
            matching_point.z = states[i].position.z;
            line_strip.points.push_back(p_current);
            line_strip.points.push_back(matching_point);
            closingLine_pub.publish(line_strip);


        }
    }
    int frameCount;

private: 
    LoopClosingTool* loopDetector_;
    ros::NodeHandle* nh_;
    ros::Subscriber keyframeSub_;
    ros::Publisher closingLine_pub;
    ros::Publisher keyframe_pub;
    ros::Publisher match_pub;
    std::vector<frontend::Keyframe> keyframes;
    std::unordered_map<int, inekf_msgs::State> states;
    inekf_msgs::State current_state_;
    int markerId;
   
    //subscriber

};
//loop closing entry
loop_closing_ros loopclosing;
void filterCallback(const inekf_msgs::StateConstPtr &stateMsg,const frontend::Keyframe::ConstPtr& Framemsg) {

    ROS_INFO("I heard: [%d],  [%d]", loopclosing.frameCount,Framemsg->frameID);
    loopclosing.run_loopClosure(Framemsg,stateMsg,loopclosing.frameCount);
    loopclosing.frameCount++;
};














int main(int argc, char **argv)
{
  ros::init(argc, argv, "loop_closing");
  ros::NodeHandle nh;
  fbow::Vocabulary voc;
  voc.readFromFile("/root/ws/curly_slam/catkin_ws/sift.fbow");
  LoopClosingTool lct(&voc);
  //set up loop closing
  loopclosing.set_core(&nh,&lct);
  //get parameters
  string keyframe_topic;
  string state_topic;
  nh.param<string>("depth_topic",  keyframe_topic, "/frontend/keyframe");
  nh.param<string>("state_topic", state_topic,  "/cheetah/inekf_estimation/inekf_state");
  //message filters
  message_filters::Subscriber<frontend::Keyframe> keyframe_sub_(nh, keyframe_topic, 1);
  message_filters::Subscriber<inekf_msgs::State> state_sub_(nh, state_topic, 5000);
  typedef message_filters::sync_policies::ApproximateTime<inekf_msgs::State, frontend::Keyframe> SyncPolicy;
  message_filters::Synchronizer<SyncPolicy> sync(SyncPolicy(5000), state_sub_,  keyframe_sub_);
  sync.registerCallback(boost::bind(&filterCallback, _1, _2));
//message_filters::Subscriber<sensor_msgs::msg::Image> disparity_sub_;
  ros::spin();

  return 0;
}