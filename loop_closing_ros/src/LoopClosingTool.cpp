#include "LoopClosingTool.hpp"
#include <algorithm>
#include <opencv2/xfeatures2d.hpp>
LoopClosingTool::LoopClosingTool(fbow::Vocabulary* pDB,shared_ptr<Camera> camera,shared_ptr<Config> config):pDB_(pDB),
                                                                                                            camera_(camera),
                                                                                                            config_(config){   
        camera_mat= (cv::Mat_<double>(3, 3) << parameter.FX, 0., parameter.CX, 0., parameter.FY, parameter.CY, 0., 0., 1.);
        lastLoopClosure_ = -1;
        currentGlobalKeyframeId = 0;
    }

bool LoopClosingTool::detect_loop(vector<Matchdata>& point_matches,std::unordered_map<int, inekf_msgs::State>& states){
    
    if (lastLoopClosure_ != -1 && currentGlobalKeyframeId - lastLoopClosure_ < skip_frame_ ){
        return false;
    }
    //first add new key frame in 
    descriptors.push_back(currentDescriptors);
    goodKeypoints = currentKeypoints;
    generateKeyframe();
    max_loopClosureId = -1;
    //find match of current frame
    int candidate_id;
    Matchdata point_match;
    bool loop_detected = find_connection(keyframes_[currentGlobalKeyframeId],candidate_id,point_match,states);
    if(loop_detected){
        point_matches.push_back(point_match);
    }
    
    return loop_detected;
}
bool LoopClosingTool::find_connection(Keyframe& frame,int& candidate_id,Matchdata& point_match,std::unordered_map<int, inekf_msgs::State>& states){
    class Compare_score{
        public:
        bool operator() (pair<int,double>& a, pair<int,double>& b) {
            return a.second < b.second;
        }
    };
    //create a temporal current keyframe
    cv::Mat cur_desc = frame.descriptors;
    cv::Mat img = frame.img;;
    int maxId = std::max(int(pDB_->size() - first_candidate_),0);
    int top = top_match_;
    std::priority_queue<pair<int,double>, std::vector<pair<int,double>>,Compare_score> pq;
    bool loop_detected = false;
    for (int i = 0; i < (int(frame.globalKeyframeID) - int(near_frame_)); i ++ ){
        fbow::fBow bowvector_cur;
        if (keyframes_[i].globalKeyframeID  < 0) continue;
        // if(keyframes_[i].descriptors.empty()){
        //     //ROS_ERROR_STREAM("size" << keyframes_.size());
        //     ROS_DEBUG_STREAM(keyframes_[i].globalKeyframeID << " " << i  << "empty old descriptor");
        //     continue;
        // }
        bowvector_cur = pDB_->transform(cur_desc);
        fbow::fBow bowvector_old;
        bowvector_old = pDB_->transform(keyframes_[i].descriptors);
        double score = fbow::fBow::score(bowvector_cur,bowvector_old);
        pq.push( std::make_pair (i, score));
    }
    //calculate score of prev frame 
    double prevScore = 1;
    if(keyframes_.count(frame.globalKeyframeID -1)!= 0){
        fbow::fBow bowvector_prev = pDB_->transform(keyframes_[frame.globalKeyframeID -1].descriptors);
        fbow::fBow bowvector_cur = pDB_->transform(cur_desc);
        prevScore = fbow::fBow::score(bowvector_cur,bowvector_prev);
        ROS_DEBUG_STREAM("prevScore is "<< prevScore);
    }
    // simple logic check to filter out unwanted
    if (pq.empty()) {
        return false;
    }
    //make sure closet frame have a good score
    int Maxinlier_Id = INT_MIN;
    RelativePose Maxinlier_Pose;
    int Maxinlier = INT_MIN;
    // Store retured match
    vector<cv::DMatch> returned_matches;
    if (pq.size() > 0){
        for (int i = 0; i < top && !pq.empty() ; i ++ ){
            int candidate_id = pq.top().first;
            double candidate_score = pq.top().second;
           // if (candidate_id )
            pq.pop();
        //     DBoW3::Result r = rets[i];
        //     // if (abs(int(r.Id) - int(rets[i-1].Id)) < 3 ){
        //     //     continue;
        //     // }
        
            if (candidate_score <= prevScore) {
        // pDB_->addImg(img);
        // //histKFs_.push_back(kf);
        // //std::cout << "added img\n";
        // return false;
            continue;
            } 
            // if (keyframes_[candidate_id].descriptors.empty()){
            //     continue;
            // }
            //calculate relative pose
            // auto current_state = current_state;
             ROS_DEBUG_STREAM("candidate_id "<< candidate_id << "  score::  " << candidate_score);
            if (states.find(candidate_id) == states.end()){
                ROS_DEBUG_STREAM("cannot find candidate state");
            }
            if (states.find(frame.globalKeyframeID) == states.end()){
                ROS_DEBUG_STREAM("cannot find current state");
            }
            auto old_state = states[candidate_id];
            auto current_state = states[frame.globalKeyframeID];
            Eigen::Vector3f    position_cur(current_state.position.x, current_state.position.y, current_state.position.z);
            Eigen::Quaternionf poseOrientation_cur(current_state.orientation.w, current_state.orientation.x, current_state.orientation.y, current_state.orientation.z);
            Sophus::SE3f currentInekfPose(poseOrientation_cur,position_cur);

            Eigen::Vector3f    position_old(old_state.position.x, old_state.position.y, old_state.position.z);
            Eigen::Quaternionf poseOrientation_old(old_state.orientation.w, old_state.orientation.x, old_state.orientation.y, old_state.orientation.z);
            Sophus::SE3f oldInekfPose(poseOrientation_old,position_old);

            Sophus::SE3f relativePose = oldInekfPose.inverse() * currentInekfPose;
            RelativePose pose( relativePose.translation(),relativePose.rotationMatrix());
            Keyframe candidate_frame;
            if (keyframes_.count(candidate_id) != 0){
                candidate_frame = keyframes_[candidate_id];
            }else{
                ROS_ERROR_STREAM("cannot find frame " << candidate_id);
            }
            int inlier = ransac_featureMatching(frame,candidate_frame);
            eliminateOutliersPnP(frame,candidate_frame,pose);
            //eliminateOutliersFundamental(frame,candidate_frame);
            //ransac_matches = good_matches;
            //eliminateOutliersFundamental(frame,candidate_frame);
            inlier = ransac_matches.size();
            //int inlier = 100;
            if (inlier > inlier_){
                loop_detected = true;
                if (inlier >  Maxinlier){
                    returned_matches.assign(ransac_matches.begin(), ransac_matches.end());
                    Maxinlier_Id = candidate_id;
                    Maxinlier = inlier;
                    Maxinlier_Pose = pose;
                }            
            }
            good_matches.clear();
            ransac_matches.clear();
      }
    }else{
       loop_detected = false; 
    }
    //pDB_->add(cur_desc);
    
    if (loop_detected){
        lastLoopClosure_ = currentGlobalKeyframeId;
       point_match = genearteNewGlobalId(frame,keyframes_[Maxinlier_Id],returned_matches,Maxinlier_Pose);
    }
    return loop_detected;
}

void LoopClosingTool::eliminateOutliersFundamental(Keyframe& current,Keyframe& candidate){
    // vector<uchar> status;
    // vector<cv::Point2f> lastPoints;
    // vector<cv::Point2f> currentPoints;
    // for (auto kp: good_lastKeypoints){
    //     lastPoints.push_back(kp.pt);
    // }
    // for (auto kp: goodKeypoints){
    //     currentPoints.push_back(kp.pt);
    // }
    // ransac_matches.clear();
    // cv::findFundamentalMat(lastPoints, currentPoints, cv::FM_RANSAC, 3.0, 0.99, status);
    // for (int i = 0; i < lastPoints.size(); i++) {
    //     if (status[i]) {
    //         ransac_matches.push_back(good_matches[i]);
    //         ransac_matches_id_map.insert({good_matches[i].trainIdx, good_matches[i].queryIdx});
    //     }
    // }
    cv::Mat imMatches;
    cv::drawMatches(candidate.img, candidate.keypoints, current.img, current.keypoints, ransac_matches, imMatches, cv::Scalar(0, 0, 255), cv::Scalar::all(-1));
            //cv::imshow("matches_window", imMatches);
            cv::namedWindow("image", cv::WINDOW_AUTOSIZE);
            cv::imshow("image", imMatches);
            //if(ransac_matches.size() > 2){
                 cv::imwrite("/home/bigby/ws/catkin_ws/result" + std::to_string(id)+ ".bmp",imMatches );
            //}
            //cv::drawMatches(lastImage, lastKeypoints, currentImage, currentKeypoints, ransac_matches, imMatches, cv::Scalar(0, 0, 255), cv::Scalar::all(-1));
            //cv::imshow("matches_window", imMatches);
            //cv::waitKey(1);
            cv::waitKey(1);
            ROS_DEBUG_STREAM("Total match " << ransac_matches.size() );
}

int LoopClosingTool::ransac_featureMatching(Keyframe& current,Keyframe& candidate){
    //clear previous matches 
    good_matches.clear();
    goodKeypoints.clear();
    good_lastKeypoints.clear();
    good_lastpoint3d.clear();
    cv::Mat cur_descr,candidate_descr;
    std::vector<cv::KeyPoint> cur_keypoints, candidate_keypoints;
    cur_descr = current.descriptors;
    candidate_descr = candidate.descriptors;
    cur_keypoints =current.keypoints;
    candidate_keypoints = candidate.keypoints;
    cv::Mat curImg = current.img;
    cv::Mat candidateImg = candidate.img;
    //create a matcher (Flann ) 
    // matcher.match( cur_descr, candidate_descr, matches );
    //mathcer brute force
    cv::BFMatcher matcher( cv::NORM_HAMMING,true);
    std::vector<cv::DMatch> matches;
    matcher.match( candidate_descr,cur_descr, matches );
    cv::Mat match_img;

    //find good match
    //if a distance is bigger then 2*min distance, we assume is false
    //to avoid extrme small mindistance we use 30 
    std::vector<cv::DMatch> normalMatches;
    // find the largest and smallest feature distances
    double min_dist = 10000, max_dist = 0;
    for (int i = 0; i < matches.size(); i++) {
        double dist = matches[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    for (int i = 0; i < matches.size(); i++) {
        if (matches[i].distance <= 2* max(min_dist,5.0)) {
            normalMatches.push_back(matches[i]);
        }
    }


    // std::vector<cv::Point2f> curKps, candKps;
    //     for (int i = 0; i < normalMatches.size(); i++) {
    //         candKps.push_back( candidate_keypoints[normalMatches[i].trainIdx].pt);
    //         curKps.push_back(cur_keypoints[normalMatches[i].queryIdx].pt);
    // }
  
    // // Use ransac to further remove outlier
    // cv::Mat H;  // cv::Mat H;
    // H = cv::findHomography(cv::Mat(curKps),cv::Mat(candKps),cv::RANSAC,5);
    // //check if H is empty
    // if(H.empty()){return 0;}
    // cv::Mat curTransformed;
    // cv::perspectiveTransform(cv::Mat(curKps),curTransformed,H);
    // // save inlier
    // for (int i =0; i < normalMatches.size(); i ++){
    //     if (cv::norm(candKps[i] - curTransformed.at<cv::Point2f>((int)i,0)) <= 5){
    //         good_matches.push_back(normalMatches[i]);
    //     }
    // }
 
    // good_matches = matches;
    // auto candidate_3dpoints = candidate.point_3d;
    // for (int i = 0; i < good_matches.size(); i++) {
    //         goodKeypoints.push_back(cur_keypoints[good_matches[i].trainIdx]);
    //         good_lastKeypoints.push_back(candidate_keypoints[good_matches[i].queryIdx]);
    //         good_lastpoint3d.push_back(candidate_3dpoints[good_matches[i].queryIdx]);
    // }
    good_matches = normalMatches;
    return good_matches.size();
}
void LoopClosingTool::create_feature(){
    currentKeypoints.clear();
    currentDescriptors.release();
    cv::Ptr<cv::FeatureDetector> detector;
    switch (featureType_) {
    case 1:
           detector = cv::ORB::create(featureCount_);
           break;
    case 2:
    #ifdef COMPILE_WITH_SURF
            detector = cv::xfeatures2d::SURF::create(featureCount_);
    #else
           throw std::runtime_error("Surf not compiled");
    #endif
            break;
        case 3:
    #ifdef COMPILE_WITH_SIFT
            detector = cv::SIFT::create(featureCount_);
    #else
            throw std::runtime_error("Sift not compiled");
    #endif-6, 
            break;
        case 4:
            detector = cv::KAZE::create();
            break;
        case 5:
            detector = cv::AKAZE::create();
            break;
        }
    //descriptor->compute(currentImage, currentKeypoints, currentDescriptors);
    detector->compute(currentImage, currentKeypoints, currentDescriptors);
}
void LoopClosingTool::create_feature(std::vector<cv::KeyPoint> Keypoints){
    currentKeypoints.clear();
    currentDescriptors.release();
    currentKeypoints = Keypoints;
    cv::Ptr<cv::FeatureDetector> detector;
    if(Keypoints.empty()){
        cout <<"keypoint is empty" << endl;
    }
    switch (featureType_) {
    case 0:
           detector = cv::ORB::create(1000, 1.2,4);
           break;
    case 1:
    #ifdef COMPILE_WITH_SURF
            detector = cv::xfeatures2d::SURF::create(featureCount_);
    #else
           throw std::runtime_error("Surf not compiled");
    #endif
            break;
        case 2:
            detector = cv::SIFT::create();
            break;
        case 3:
            detector = cv::KAZE::create();
            break;
        case 4:
            detector = cv::AKAZE::create();
            break;
        }
    detector->compute(currentImage, Keypoints,currentDescriptors);
    
    
}
void LoopClosingTool::assignNewFrame(const cv::Mat &img,const cv::Mat &depth,int gloablKeyframeId,std::vector<int> globalID){
    currentImage = img;
    currentDepth = depth;
    currentGlobalKeyframeId =  gloablKeyframeId;
    //currentGlobalKeyframeId++;
    current_globalIDs = globalID;

}
void LoopClosingTool::generateKeyframe(){
    //calculate point 3d ]
    //get3DfeaturePosition(point_3d, currentDepth, goodKeypoints);
    if (currentDescriptors.empty()){
        ROS_ERROR_STREAM("empty descriptor");
        exit(1);
    }
    if (currentDescriptors.empty()){
            ROS_ERROR_STREAM("empty current descriptor");
            exit(1);
    }   
   
    Keyframe kf(currentGlobalKeyframeId,currentImage,currentDepth,currentKeypoints,currentDescriptors);
    kf.insertPoint3D(point_3d);
    kf.insertGlobalID(current_globalIDs);
    if(keyframes_.count(currentGlobalKeyframeId) ==0 ){
        keyframes_[currentGlobalKeyframeId] = kf;
    }else{
        ROS_ERROR_STREAM("Redundant keyframe detected ");
    }
    
    goodKeypoints.clear();
    currentKeypoints.clear();
    
   
}
void LoopClosingTool::get2DfeaturePosition(vector<cv::Point2f> &point_2d, const vector<cv::KeyPoint> &good_kp2){
    point_2d.clear();
    for (size_t i = 0; i < good_kp2.size(); ++i) {
        point_2d.push_back(good_kp2[i].pt);
    }
}
void LoopClosingTool::get3DfeaturePosition(vector<cv::Point3f> &point_3d, const cv::Mat &dpt1, const vector<cv::KeyPoint> &good_kp1) {

    point_3d.clear();
    for (size_t i = 0; i < good_kp1.size(); ++i) {
        int          u = good_kp1[i].pt.x;
        int          v = good_kp1[i].pt.y;
        unsigned int d = dpt1.ptr<unsigned short>(v)[u];

        if (isnan(d)) {
            std::cout << "nan" << std::endl;
            continue;
        }
        float x = double(d) * parameter.PIXEL_TO_METER_SCALEFACTOR;
        float y = (u - parameter.CX) * x / parameter.FX;
        float z = (v - parameter.CY) * x / parameter.FY;
        point_3d.push_back(cv::Point3f(x, y, z));
    }
}

void LoopClosingTool::assignRansacGuess(const Eigen::Matrix3f &rot, const Eigen::Vector3f &pos){
    cv::eigen2cv(rot, ransacRGuess);
    cv::eigen2cv(pos, ransacTGuess);
}

void LoopClosingTool::eliminateOutliersPnP(Keyframe& current,Keyframe& candidate, RelativePose& pose){
    cout <<"start loop closure pnp " << endl;
    ransac_matches.clear();
    vector<cv::Point3f> candidate_3d = good_lastpoint3d; //3d point from candidate
    //get3DfeaturePosition(candidate_3d, candidate.depth,good_lastKeypoints);
    //candidate_3d = candidate.point_3d;
    get2DfeaturePosition(point_2d,goodKeypoints);
    //ransac guess
    assignRansacGuess(pose.rot,pose.pos);
    if (candidate_3d.size() >= 4) {
        cv::Mat inliers, ransacRVectorGuess;
        Rodrigues(ransacRGuess, ransacRVectorGuess);
        //convert point 3d 2d size to same size
        vector<cv::Point2f> point_2d_use;
        vector<cv::Point3f> point_3d_use;
        // if (point_2d.size() >=  candidate_3d.size()){
        //     point_3d_use = candidate_3d;
        //     for(int i = 0 ; i <  candidate_3d.size() ;i ++){
        //         point_2d_use.push_back(point_2d[i]);
        //     }
        // }else {
        //     point_2d_use = point_2d;
        //     for(int i = 0 ; i <  point_2d.size() ;i ++){
        //         point_3d_use.push_back(candidate_3d[i]);
        //     }
        // }
        cv::solvePnPRansac(candidate_3d,
                           point_2d,
                           camera_->K_cv(),
                           distort,
                           ransacRVectorGuess,
                           ransacTGuess,
                           true,
                           ransacIterations_,
                           ransacReprojectionError_,
                           0.99,
                           inliers,
                           cv::SOLVEPNP_ITERATIVE);

        ransac_matches.clear();
        ransac_matches_id_map.clear();
        for (size_t i = 0; i < inliers.rows; ++i) {
            ransac_matches.push_back(good_matches[inliers.at<int>(i, 0)]);
            ransac_matches_id_map.insert({good_matches[inliers.at<int>(i, 0)].trainIdx, good_matches[inliers.at<int>(i, 0)].queryIdx});
        }
        Rodrigues(ransacRVectorGuess,ransacRGuess);
    } else {
        ransac_matches = good_matches;
        for (size_t i = 0; i < ransac_matches.size(); ++i) {
            ransac_matches_id_map.insert({good_matches[i].trainIdx, good_matches[i].queryIdx});
        }
    }
    cv::Mat lastImage = candidate.img;
    std::vector<cv::KeyPoint> lastKeypoints = candidate.keypoints;
    cv::Mat imMatches;

    // get optimized pose 
    
    cv::cv2eigen(ransacTGuess,pose.pos);
    Eigen::Matrix3f rot;
    cv::cv2eigen(ransacRGuess,pose.rot);
    
    id++;
     try {
        if (ransac_matches.size() > inlier_){
            cv::drawMatches(lastImage, lastKeypoints, current.img, current.keypoints, ransac_matches, imMatches, cv::Scalar(0, 0, 255), cv::Scalar::all(-1));
            //cv::imshow("matches_window", imMatches);
            cv::namedWindow("image", cv::WINDOW_AUTOSIZE);
            cv::imshow("image", imMatches);
            //if(ransac_matches.size() > 2){
                 cv::imwrite("/home/bigby/ws/catkin_ws/result" + std::to_string(id)+ ".bmp",imMatches );
            //}
            //cv::drawMatches(lastImage, lastKeypoints, currentImage, currentKeypoints, ransac_matches, imMatches, cv::Scalar(0, 0, 255), cv::Scalar::all(-1));
            //cv::imshow("matches_window", imMatches);
            //cv::waitKey(1);
            cv::waitKey(1);
            ROS_DEBUG_STREAM("Total match " << ransac_matches.size() );
        }
        
    } catch (...) {
       ROS_ERROR_STREAM("failed to plot");
    }
    cout << "loopclosure pnp match size: " << good_matches.size() << "," << ransac_matches.size() << endl;

   
}
Matchdata LoopClosingTool::genearteNewGlobalId(Keyframe& current, Keyframe& candidate,vector<cv::DMatch>& returned_matches,RelativePose& pose){
    std::vector<int> candidate_globalId = candidate.globalIDs;
    //check ransac matches, find matched global id, created map
    std::unordered_map<int,int> matched_globalId; 
    std::vector<int> current_globalIDs_ = current.globalIDs;
    std::vector<int> cur_pointId;
    std::vector<int> old_pointId;
    std::vector<cv::KeyPoint> newmeasurement;
    for (int i = 0; i < returned_matches.size(); i ++){
        cur_pointId.push_back( current_globalIDs_[returned_matches[i].trainIdx]);
        old_pointId.push_back( candidate_globalId[returned_matches[i].queryIdx]);
        newmeasurement.emplace_back(current.keypoints[returned_matches[i].trainIdx]);
    }
    Matchdata point_match(current.globalKeyframeID,candidate.globalKeyframeID,cur_pointId,old_pointId,newmeasurement,pose);
    
    //current_globalIDs = result_globalId;
    // return result_globalId;
    return point_match;
}