#include "ros/ros.h"
#include "LoopClosingTool.hpp"
#include "std_msgs/String.h"
#include "frontend/Keyframe.h"
#include "string"
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <inekf_msgs/State.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/Point.h>
#include <unordered_map>
class loop_closing_ros{
public:
    loop_closing_ros(){markerId = 0;}
    
    //set up loop closure tool
    void set_core(ros::NodeHandle* nh, LoopClosingTool* ltr){
        nh_ = nh;
        loopDetector_ = ltr;
        closingLine_pub = nh->advertise<visualization_msgs::Marker>("loop_closing_line", 10);
    }
    void set_keyframe(const frontend::Keyframe::ConstPtr& msg){
        keyframes.push_back(*msg);
        cv_bridge::CvImagePtr color_ptr;
        cv_bridge::CvImagePtr depth_ptr;
        sensor_msgs::ImageConstPtr colorImg( new sensor_msgs::Image( msg->color ) );
        sensor_msgs::ImageConstPtr depthImg( new sensor_msgs::Image( msg->depth ) );
        try {
            color_ptr = cv_bridge::toCvCopy(colorImg, sensor_msgs::image_encodings::BGR8);
            depth_ptr = cv_bridge::toCvCopy(depthImg, sensor_msgs::image_encodings::TYPE_16UC1);
        } catch (cv_bridge::Exception &e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
        cv::Mat color = color_ptr->image;
        cv::Mat depth = depth_ptr->image;
        //IC(color);
        cv::imshow("dispaly",color);
        vector<int> matchingIndex;
        loopDetector_->assignNewFrame(color,depth,msg->frameID);
        loopDetector_->create_feature();
        loopDetector_->detect_loop(matchingIndex);
        if (!matchingIndex.empty()){
            draw_line(matchingIndex);
        }
        IC(matchingIndex);

    }
    void set_state(const inekf_msgs::StateConstPtr &stateMsg,int frameID){
        current_state_ = (*stateMsg);
        states[frameID] = (*stateMsg);
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
private: 
    LoopClosingTool* loopDetector_;
    ros::NodeHandle* nh_;
    ros::Subscriber keyframeSub_;
    ros::Publisher closingLine_pub;
    std::vector<frontend::Keyframe> keyframes;
    std::unordered_map<int, inekf_msgs::State> states;
    inekf_msgs::State current_state_;
    int markerId;

    //subscriber

};
//loop closing entry
loop_closing_ros loopclosing;
void filterCallback(const inekf_msgs::StateConstPtr &stateMsg,const frontend::Keyframe::ConstPtr& Framemsg) {

    loopclosing.set_keyframe(Framemsg);
    loopclosing.set_state(stateMsg,Framemsg->frameID);
    ROS_INFO("I heard: [%d]", Framemsg->frameID);
};



























int main(int argc, char **argv)
{
  ros::init(argc, argv, "loop_closing");
  ros::NodeHandle nh;
  DBoW3::Vocabulary voc("/root/ws/curly_slam/catkin_ws/src/loop_closing_slam/data/orbvoc.dbow3");
  DBoW3::Database db(voc, false, 0);
  LoopClosingTool lct(&db);
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