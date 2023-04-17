#include "LoopClosingManager.hpp"
void LoopClosingManager::setCore(ros::NodeHandle* nh, LoopClosingTool* ltr){
    nh = nh;
    loopDetector = ltr;
    closingLine_pub = nh->advertise<visualization_msgs::Marker>("loop_closing_line", 10);
    test_point_pub = nh->advertise<visualization_msgs::Marker>("loop_closing_pose", 10);
    lc_pointcloud_pub = nh->advertise<visualization_msgs::MarkerArray>("loop_closing_point", 10);
    //keyframe_pub = nh->advertise<frontend::Keyframe>("loop_closing/keyframe", 10);
    match_pub = nh->advertise<frontend::Match>("match", 10);
    frameCount = 0;
}
void LoopClosingManager::runLoopClosure(const frontend::Keyframe::ConstPtr& msg,const inekf_msgs::StateConstPtr &stateMsg){
    keyframes.push_back(*msg);
    current_state = (*stateMsg);
    states[msg->frameID] = (*stateMsg);
    if (!cameraIntialized){
        ROS_ERROR_STREAM("Camera not intialized");
    }
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
    ROS_DEBUG_STREAM("load descriptor"); 
    
    cv::Mat descriptor= cv::Mat::zeros(msg->descriptor_shape[0], msg->descriptor_shape[1], CV_32F);
    for (int i = 0; i < msg->descriptor_shape[0]; i ++){
        for (int j = 0; j < msg->descriptor_shape[1];j++){
            descriptor.at<float>(i,j) = msg->descriptor[i * msg->descriptor_shape[1] + j];
        }
    }
     ROS_DEBUG_STREAM("load descriptor Complete"); 
    //cv::Mat descriptor = descriptor_ptr->image;
    
    //extract globalid, and key points, 
    std::vector<unsigned int> globalId;
    std::vector<cv::Point2f> feature_2d;
    std::vector<cv::Point3f> feature_3d;
    std::vector<cv::KeyPoint> keypoints;
    for (int i = 0; i < msg-> features.size(); i++){
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
    vector<int> matchingIndex;
    loopDetector->assignNewFrame(color,depth,msg->frameID,globalId);
    loopDetector->create_feature(keypoints,descriptor);
    //loopDetector->create_feature();
    // loopDetector->set2DfeaturePosition(feature_2d);  
    loopDetector->set3DfeaturePosition(feature_3d);
    vector<int> connections;
    for (auto id:msg->connection){
        connections.push_back(id);
    }
    //loopDetector->assignRansacGuess(poseOrientation.toRotationMatrix(),positionVector);
    vector<Matchdata> point_matches;
    bool loopdetected = loopDetector->detect_loop(point_matches,current_state,connections);
    //cv::imwrite("image"+std::to_string(keyframes.size())+".jpg",color );
    if (loopdetected){
        //update globalId
        drawLine(point_matches[point_matches.size() -1].oldId_);
        ROS_INFO_STREAM(point_matches.empty() <<" xxxxloop_detected between " << point_matches[point_matches.size() -1].oldId_ << " and " << point_matches[point_matches.size() -1].curId_);
        publishMatch(point_matches[point_matches.size() -1]);
        //publish Matched data
    }
    // }else{
    //    // keyframe_pub.publish(*msg);
    // }

}
void LoopClosingManager::publishMatch(Matchdata& point_match){
    cout << point_match.curId_ <<" ---x-- " << point_match.oldId_ << endl;
    frontend::Match match_msg;
    if(point_match.point_current_.size() == 0){
        match_pub.publish(match_msg);
        return;
    }
    match_msg.curId = point_match.curId_;
    match_msg.oldId = point_match.oldId_;
    match_msg.curPoint = point_match.point_current_;
    match_msg.oldPoint = point_match.point_old_;
    ROS_INFO_STREAM("curId " << match_msg.curPoint.size() << " oldId " << match_msg.oldPoint.size() << endl);
    vector<geometry_msgs::Point> measurement;
    for (auto keypoint:point_match.newmeasurement_){
        geometry_msgs::Point newPoint;
        newPoint.x = keypoint.pt.x;
        newPoint.y = keypoint.pt.y;
        measurement.push_back(newPoint);
    }
    match_msg.measurement = measurement;

    //test relative pose calculate from pnp, by add this relative pose to old pose
    auto old_state = states[match_msg.oldId];
    Eigen::Vector3f    position_old(old_state.position.x, old_state.position.y, old_state.position.z);
    Eigen::Quaternionf poseOrientation_old(old_state.orientation.w, old_state.orientation.x, old_state.orientation.y, old_state.orientation.z);
    Sophus::SE3f oldInekfPose(poseOrientation_old,position_old);
    Eigen::Quaternionf poseOrientation_relative(point_match.rp_.rot);
    Sophus::SE3f relativePose(poseOrientation_relative,point_match.rp_.pos);
     
    
    Sophus::SE3f currentPose_estimate =  relativePose * oldInekfPose;
    auto t = relativePose.translation();
    auto q = relativePose.unit_quaternion().coeffs();
    geometry_msgs::Pose estimate_pose;
    estimate_pose.orientation.w = q.w();
    estimate_pose.orientation.x = q.x();
    estimate_pose.orientation.y = q.y();
    estimate_pose.orientation.z = q.z();

    estimate_pose.position.x = t.x();
    estimate_pose.position.y = t.y();
    estimate_pose.position.z = t.z();
    // currentPose_estimate = camera->CameraTrajectoryToImu(currentPose_estimate);
    drawPoint(currentPose_estimate);
    match_msg.betweenPose = estimate_pose;
    // auto current_state =  states[match_msg.curId];
    // auto old_state = states[match_msg.oldId];
    // Eigen::Vector3f    position_cur(current_state.position.x, current_state.position.y, current_state.position.z);
    // Eigen::Quaternionf poseOrientation_cur(current_state.orientation.w, current_state.orientation.x, current_state.orientation.y, current_state.orientation.z);
    // Sophus::SE3f currentInekfPose(poseOrientation_cur,position_cur);

    // Eigen::Vector3f    position_old(old_state.position.x, old_state.position.y, old_state.position.z);
    // Eigen::Quaternionf poseOrientation_old(old_state.orientation.w, old_state.orientation.x, old_state.orientation.y, old_state.orientation.z);
    // Sophus::SE3f oldInekfPose(poseOrientation_old,position_old);

    // Sophus::SE3f relativePose = oldInekfPose.inverse() * currentInekfPose;
    // auto t = relativePose.translation();
    // auto q = relativePose.unit_quaternion().coeffs();

    // geometry_msgs::Pose betweenPose;
    // betweenPose.orientation.w = q.w();
    // betweenPose.orientation.x = q.x();
    // betweenPose.orientation.y = q.y();
    // betweenPose.orientation.z = q.z();

    // betweenPose.position.x = t.x();
    // betweenPose.position.y = t.y();
    // betweenPose.position.z = t.z();

    // match_msg.betweenPose = betweenPose;
    
    
    match_pub.publish(match_msg);
}
void LoopClosingManager::drawLine(int i){
    geometry_msgs::Point p_current;
    p_current.x = current_state.position.x;
    p_current.y = current_state.position.y;
    p_current.z = current_state.position.z;
    
    visualization_msgs::Marker line_strip;
    line_strip.header.frame_id = config->worldFrame;
    line_strip.header.stamp = ros::Time::now();
    line_strip.ns =  "points_and_lines";
    line_strip.action = visualization_msgs::Marker::ADD;
    line_strip.type = visualization_msgs::Marker::LINE_STRIP;
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
void LoopClosingManager::drawLine(const vector<int>& matchingIndex){
    geometry_msgs::Point p_current;
    p_current.x = current_state.position.x;
    p_current.y = current_state.position.y;
    p_current.z = current_state.position.z;
    for(int i = 0; i < matchingIndex.size();i++){
        visualization_msgs::Marker line_strip;
        line_strip.header.frame_id = config->worldFrame;
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
void LoopClosingManager::drawPoint(const Sophus::SE3f pose){
    //publish pose
    auto t = pose.translation();
    auto q = pose.unit_quaternion().coeffs();
    ROS_DEBUG_STREAM("Publish relative pose marker");
    visualization_msgs::Marker test_point;
    test_point.header.frame_id = config->worldFrame;;
    test_point.header.stamp = ros::Time::now();
    test_point.ns =  "points";
    test_point.action = visualization_msgs::Marker::ADD;
    test_point.pose.orientation.w = q.w();
    test_point.pose.orientation.x = q.x();
    test_point.pose.orientation.y = q.y();
    test_point.pose.orientation.z = q.z();
    test_point.pose.position.x = t.x();
    test_point.pose.position.y= t.y();
    test_point.pose.position.z = t.z();
    test_point.id = 0;
    test_point.color.b = 1.0;
    test_point.color.r = 0.0;
    test_point.color.a = 1.0;
    test_point.scale.x = 0.5;
    test_point.scale.y = 0.1;
    test_point.scale.z = 0.1;
    test_point.type = visualization_msgs::Marker::ARROW;
  
    test_point_pub.publish(test_point);

    visualization_msgs::MarkerArray inViewPointArray;
    int id = 1;
    //publish point 
    for (auto pair:loopDetector->loopClosurePoint){
        int status = pair.second;
        auto landmark = pair.first;
        visualization_msgs::Marker newPoint;
        newPoint.header.frame_id = config->worldFrame;;
        newPoint.header.stamp = ros::Time::now();
        newPoint.ns =  "Inview_point";
        newPoint.action = visualization_msgs::Marker::ADD;
        newPoint.pose.orientation.w = q.w();
        newPoint.pose.orientation.x = q.x();
        newPoint.pose.orientation.y = q.y();
        newPoint.pose.orientation.z = q.z();
        newPoint.pose.position.x = landmark->pointGlobal[0];
        newPoint.pose.position.y= landmark->pointGlobal[1];
        newPoint.pose.position.z = landmark->pointGlobal[2];
        newPoint.id = id++;
           if (status == 2){
                // blue for matched point 
                newPoint.color.b = 0.5;
                newPoint.color.g = 0.0;
                newPoint.color.r = 0.0;
           }else if (status == 3){
                // green for point observed by current frame
                newPoint.color.b = 0.0;
                newPoint.color.g = 0.5;
                newPoint.color.r = 0.0;
           }else if (status == 4){
                // yellow for overlapping
                newPoint.color.b = 0/255;
                newPoint.color.g = 255/255;
                newPoint.color.r = 255/255;
           }
           else{
                //  red for inview point
                newPoint.color.b = 127/255;
                newPoint.color.g = 0.0;
                newPoint.color.r = 255/255;
           }
        newPoint.color.a = 1.0;
        newPoint.scale.x = 0.1;
        newPoint.scale.y = 0.1;
        newPoint.scale.z = 0.1;
        newPoint.type = visualization_msgs::Marker::SPHERE;
        inViewPointArray.markers.push_back(newPoint);
    }
    lc_pointcloud_pub.publish(inViewPointArray);
}