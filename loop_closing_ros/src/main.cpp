#include "ros/ros.h"
#include "LoopClosingTool.hpp"
#include "std_msgs/String.h"
#include "frontend/Keyframe.h"
#include "string"
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
class loop_closing_ros{
public:
    loop_closing_ros(DBoW3::Database* pDB,ros::NodeHandle* n,LoopClosingTool* lct):pDB_(pDB),nh_(n){
        loopDetector_ = lct;
        keyframeSub_ = nh_->subscribe("/frontend/keyframe", 1000, &loop_closing_ros::keyFrameCallback, this);

    }
     void keyFrameCallback(const frontend::Keyframe::ConstPtr& msg)
    {   
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
        loopDetector_->assignNewFrame(color);
        loopDetector_->create_feature();
        loopDetector_->detect_loop();


        cv::waitKey(10);
    }
   

private: 
    DBoW3::Database* pDB_;
    LoopClosingTool* loopDetector_;
    ros::NodeHandle* nh_;
    ros::Subscriber keyframeSub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> image_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> disparity_sub_;
    //subscriber

};
int main(int argc, char **argv)
{
  ros::init(argc, argv, "loop_closing");
  ros::NodeHandle n;
  DBoW3::Vocabulary voc("/root/ws/curly_slam/catkin_ws/src/loop_closing_slam/data/orbvoc.dbow3");
  DBoW3::Database db(voc, false, 0);
  Eigen::MatrixXd loopClosureMatch = Eigen::MatrixXd::Zero(3000,3000);
  LoopClosingTool lct(&db,&loopClosureMatch);
  loop_closing_ros loopclosing(&db,&n,&lct);
  ros::spin();

  return 0;
}