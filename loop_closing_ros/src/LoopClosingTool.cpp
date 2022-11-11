#include "LoopClosingTool.hpp"
#include <algorithm>
#include <opencv2/xfeatures2d.hpp>
LoopClosingTool::LoopClosingTool(fbow::Vocabulary* pDB,shared_ptr<Camera> camera,shared_ptr<Config> config):pDB_(pDB),
                                                                                                            camera_(camera),
                                                                                                            config_(config){   
        camera_mat= (cv::Mat_<double>(3, 3) << parameter.FX, 0., parameter.CX, 0., parameter.FY, parameter.CY, 0., 0., 1.);
        lastLoopClosure_ = -1;
        currentGlobalKeyframeId = 0;
        landmark_manager.reset(new LandmarkManager(config,camera));
    }

bool LoopClosingTool::detect_loop(vector<Matchdata>& point_matches,inekf_msgs::State state,vector<int>& connectedFrames){
    if (states.count(currentGlobalKeyframeId) == 0){
        states[currentGlobalKeyframeId]= state;
    }
    if (lastLoopClosure_ != -1 && currentGlobalKeyframeId - lastLoopClosure_ < skip_frame_ ){
        return false;
    }
    //first add new key frame in 
    descriptors.push_back(currentDescriptors);
    goodKeypoints = currentKeypoints;
    generateKeyframe(connectedFrames);
    max_loopClosureId = -1;
    //find match of current frame
    int candidate_id;
    Matchdata point_match;
    bool loop_detected = find_connection(keyframes_[currentGlobalKeyframeId],candidate_id,point_match);
    if(loop_detected){
        point_matches.push_back(point_match);
    }
    
    return loop_detected;
}
bool LoopClosingTool::find_connection(Keyframe& frame,int& candidate_id,Matchdata& point_match){
    class Compare_score{
        public:
        bool operator() (pair<int,double>& a, pair<int,double>& b) {
            return a.second < b.second;
        }
    };
    //create a temporal current keyframe
    cv::Mat cur_desc = frame.additionaldescriptors;
    cv::Mat img = frame.img;;
    int maxId = std::max(int(pDB_->size() - first_candidate_),0);
    int top = top_match_;
    std::priority_queue<pair<int,double>, std::vector<pair<int,double>>,Compare_score> pq;
    bool loop_detected = false;
    for (int i = 0; i < (int(frame.globalKeyframeID) - int(near_frame_)); i ++ ){
        fbow::fBow bowvector_cur;
        if (keyframes_[i].globalKeyframeID  < 0) continue;
        if(keyframes_[i].additionaldescriptors.empty()){
            ROS_ERROR_STREAM("size" << keyframes_.size());
            ROS_DEBUG_STREAM(keyframes_[i].globalKeyframeID << " " << i  << "empty old descriptor");
            continue;
        }
        bowvector_cur = pDB_->transform(cur_desc);
        fbow::fBow bowvector_old;
        bowvector_old = pDB_->transform(keyframes_[i].additionaldescriptors);
        double score = fbow::fBow::score(bowvector_cur,bowvector_old);
        pq.push( std::make_pair (i, score));
    }
    //calculate score of prev frame 
    double prevScore = 1;
    ROS_DEBUG_STREAM("start finding potential frame"); 
    
    
    if (keyframes_.count(frame.globalKeyframeID -1)!= 0 && (int(frame.globalKeyframeID) - int(near_frame_) > 0) ){
        fbow::fBow bowvector_prev = pDB_->transform(keyframes_[frame.globalKeyframeID -1].additionaldescriptors);
        fbow::fBow bowvector_cur = pDB_->transform(cur_desc);
        prevScore = fbow::fBow::score(bowvector_cur,bowvector_prev);
        ROS_DEBUG_STREAM("prevScore is "<< prevScore);
    }
    //simple logic check to filter out unwanted
    if (pq.empty()) {
        return false;
    }
    //make sure closet frame have a good score
    int Maxinlier_Id = INT_MIN;
    RelativePose Maxinlier_Pose;
    int Maxinlier = INT_MIN;
    // Store retured match
    vector<cv::DMatch> returned_matches;
    set<int> candiateFrames;
    if (pq.size() > 0){
         for (int i = 0; i < top && !pq.empty() ; i ++ ){
            int candidate_id = pq.top().first;
            double candidate_score = pq.top().second;
            pq.pop();
            if (candidate_score <= prevScore) {
                break;
            } 
            ROS_DEBUG_STREAM("candidate_id "<< candidate_id << "  score::  " << candidate_score);
            if (states.find(candidate_id) == states.end()){
                ROS_DEBUG_STREAM("cannot find candidate state");
            }
            if (states.find(frame.globalKeyframeID) == states.end()){
                ROS_DEBUG_STREAM("cannot find current state");
            }
            candiateFrames.insert(candidate_id);
            for (auto connected_id : keyframes_[candidate_id].connectedFrame){
                candiateFrames.insert(connected_id);
            }
         }
    }
    if (candiateFrames.size() > 0){
        for (auto candidate_id: candiateFrames){
            double candidate_core = pq.top().second;
        
            
          
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
                auto updatedCandidateDescriptor = landmark_manager->getDescriptors(keyframes_[candidate_id].globalIDs);
                keyframes_[candidate_id].updateDescriptors(updatedCandidateDescriptor);
                candidate_frame = keyframes_[candidate_id];
            }else{
                ROS_ERROR_STREAM("cannot find frame " << candidate_id);
            }
            int inlier = ransac_featureMatching(frame,candidate_frame);
            //eliminateOutliersPnP(frame,candidate_frame,pose);
            pnpCorrespondence(frame,candidate_frame);
            eliminateOutliersFundamental(frame,candidate_frame);
            //ransac_matches = good_matches;
            //eliminateOutliersFundamental(frame,candidate_frame);
            inlier = ransac_matches.size();
            ROS_DEBUG_STREAM("Inlier size is " << inlier);
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
    vector<uchar> status;
    vector<cv::Point2f> lastPoints;
    vector<cv::Point2f> currentPoints;
    for (auto kp: good_lastKeypoints){
        lastPoints.push_back(kp.pt);
    }
    for (auto kp: goodKeypoints){
        currentPoints.push_back(kp.pt);
    }
    ransac_matches.clear();
    if (lastPoints.size() < 8){
        return;
    }
    cv::Mat E = cv::findEssentialMat(lastPoints, currentPoints, camera_->K_cv(), cv::RANSAC, 0.999, 1.0, status);
    for (int i = 0; i < lastPoints.size(); i++) {
        if (status[i]) {
            ransac_matches.push_back(good_matches[i]);
            ransac_matches_id_map.insert({good_matches[i].trainIdx, good_matches[i].queryIdx});
        }
    }
    // cv::Mat imMatches;
    // cv::drawMatches(candidate.img, candidate.keypoints, current.img, current.keypoints, ransac_matches, imMatches, cv::Scalar(0, 0, 255), cv::Scalar::all(-1));
    //         //cv::imshow("matches_window", imMatches);
    //         cv::namedWindow("image", cv::WINDOW_AUTOSIZE);
    //         cv::imshow("image", imMatches);
    //         //if(ransac_matches.size() > 2){
    //              cv::imwrite("/home/bigby/ws/catkin_ws/result" + std::to_string(id)+ ".bmp",imMatches );
    //         //}
    //         //cv::drawMatches(lastImage, lastKeypoints, currentImage, currentKeypoints, ransac_matches, imMatches, cv::Scalar(0, 0, 255), cv::Scalar::all(-1));
    //         //cv::imshow("matches_window", imMatches);
    //         //cv::waitKey(1);
    //         cv::waitKey(1);
    //         ROS_DEBUG_STREAM("Total match " << ransac_matches.size() );
}

int LoopClosingTool::ransac_featureMatching(Keyframe& current,Keyframe& candidate){
    //clear previous matches 
    good_matches.clear();
    goodKeypoints.clear();
    good_lastKeypoints.clear();
    good_lastpoint3d.clear();
    cv::Mat cur_descr_full,candidate_descr_full;
    std::vector<cv::KeyPoint> cur_keypoints, candidate_keypoints;
    cur_descr_full = current.descriptors;
    candidate_descr_full = candidate.descriptors;
    cv::Mat cur_descr,candidate_descr;
    cur_keypoints =current.keypoints;
    candidate_keypoints = candidate.keypoints;
    for (int i = 0; i < cur_keypoints.size();i++){
        cur_descr.push_back(cur_descr_full.row(i));
    }
     for (int i = 0; i < candidate_keypoints.size();i++){
        candidate_descr.push_back(candidate_descr_full.row(i));
    }
    cv::Mat curImg = current.img;
    cv::Mat candidateImg = candidate.img;
    //create a matcher (Flann ) 
    // matcher.match( cur_descr, candidate_descr, matches );
    //mathcer brute force
    cv::BFMatcher matcher( cv::NORM_L2,true);
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
        if (matches[i].distance <= 3* max(min_dist,30.0)) {
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
 
    good_matches = normalMatches;
    // auto candidate_3dpoints = candidate.point_3d;
    vector<cv::Point3d> candidate_3dpoints;
    for (auto id: candidate.globalIDs){
        candidate_3dpoints.push_back(eigenTocv(landmark_manager->landmarks[id]->pointGlobal));
    }
    for (int i = 0; i < good_matches.size(); i++) {
            goodKeypoints.push_back(cur_keypoints[good_matches[i].trainIdx]);
            good_lastKeypoints.push_back(candidate_keypoints[good_matches[i].queryIdx]);
            good_lastpoint3d.push_back(candidate_3dpoints[good_matches[i].queryIdx]);
    }
   
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
void LoopClosingTool::create_feature(std::vector<cv::KeyPoint>& Keypoints,cv::Mat descriptors){
    currentKeypoints.clear();
    currentDescriptors.release();
    additionalDescriptors.release();
    currentKeypoints = Keypoints;
    cv::Ptr<cv::FeatureDetector> detector;
    if(Keypoints.empty()){
        cout <<"keypoint is empty" << endl;
    }
    cv::Ptr<cv::FeatureDetector>  extraDetector = cv::ORB::create(2000, 1,1);
    switch (featureType_) {
    case 0:
           detector = cv::ORB::create(2000, 1,1);
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
    // detector->compute(currentImage, Keypoints,currentDescriptors);
    currentDescriptors = descriptors;
    vector<cv::KeyPoint> additionalKeypoint;
    extraDetector->detectAndCompute(currentImage, cv::Mat(), additionalKeypoint,additionalDescriptors);
    // currentDescriptors.push_back(additionalDescriptor);
    
}
void LoopClosingTool::assignNewFrame(const cv::Mat &img,const cv::Mat &depth,int gloablKeyframeId,std::vector<int> globalID){
    currentImage = img;
    currentDepth = depth;
    currentGlobalKeyframeId =  gloablKeyframeId;
    //currentGlobalKeyframeId++;
    current_globalIDs = globalID;

}
void LoopClosingTool::generateKeyframe(vector<int>& connectedFrames){
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
   
    Keyframe kf(currentGlobalKeyframeId,currentImage,currentDepth,currentKeypoints,currentDescriptors,additionalDescriptors);
    kf.insertPoint3D(point_3d);
    kf.insertGlobalID(current_globalIDs);
    if(keyframes_.count(currentGlobalKeyframeId) ==0 ){
        keyframes_[currentGlobalKeyframeId] = kf;
    }else{
        ROS_ERROR_STREAM("Redundant keyframe detected ");
    }
    ROS_DEBUG_STREAM("recevied connection frame " << currentGlobalKeyframeId << " connections is  " << connectedFrames.size());
    for (int connectedID : connectedFrames){
        if (connectedID < currentGlobalKeyframeId)
            keyframes_[currentGlobalKeyframeId].connectedFrame.insert(connectedID);
            keyframes_[connectedID].connectedFrame.insert(currentGlobalKeyframeId);
    }
    goodKeypoints.clear();
    currentKeypoints.clear();
    
    landmark_manager->addKeyframe(kf);
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
        vector<cv::Point3f> good_ObjectPoints;
        vector<cv::Point2f> good_ImagePoints;
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
            good_ObjectPoints.push_back(candidate_3d[inliers.at<int>(i, 0)]);
            good_ImagePoints.push_back(point_2d[inliers.at<int>(i, 0)]);
        }
      
        // if (good_ObjectPoints.size() > 4){
        //     cv::solvePnPRefineLM(good_ObjectPoints,good_ImagePoints, camera_->K_cv(),distort,ransacRVectorGuess,ransacTGuess);
        // }
        // Rodrigues(ransacRVectorGuess,ransacRGuess);
    } else {
        ransac_matches = good_matches;
        for (size_t i = 0; i < ransac_matches.size(); ++i) {
            ransac_matches_id_map.insert({good_matches[i].trainIdx, good_matches[i].queryIdx});
        }
    }
    cv::Mat lastImage = candidate.img;
    std::vector<cv::KeyPoint> lastKeypoints = candidate.keypoints;
    // get optimized pose 
    
    cv::cv2eigen(ransacTGuess,pose.pos);
    Eigen::Matrix3f rot;
    cv::cv2eigen(ransacRGuess,pose.rot);
    cv::Mat imMatches;
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
void LoopClosingTool::pnpCorrespondence(Keyframe& current,Keyframe& candidate){
    processedID.clear();
    loopClosurePoint.clear();
    // traverse current frame glboal Id, prevent thouse got matched again
    for (auto Id: current.globalIDs){
        processedID[Id] = 1;
    }
    ransac_matches_id_map.clear();
    cv::Mat unmatched2dDescriptors;
    vector<cv::Point2f> unmatched2dPoint;
    unordered_map<int,int> keyPoint_index_map;
    int cnt = 0;
    auto cur_keypoints = current.keypoints;
    for (size_t i = 0; i < cur_keypoints.size(); ++i) {
        // if (ransac_matches_id_map.count(i) > 0) {
        //     // if the point is in ransac_matches, assign old ID
        //     int old_id = keyframes_[currentGlobalKeyframeId].globalIDs[ransac_matches_id_map[i]];
        //     processedID[old_id] = 1;
        // }else{
            unmatched2dDescriptors.push_back(current.descriptors.row(i));
            unmatched2dPoint.push_back(cur_keypoints[i].pt);
            keyPoint_index_map[cnt] = i;
            cnt++;
        // }
    }
   
    //get pose for current frame 
    vector<cv::Point2f> ProjectedLocations;
    
    set<int> InviewLandmarkIds;
    set<int> visited;
    getInviewPoint(InviewLandmarkIds,visited,2,candidate.globalKeyframeID);
    getInviewPoint(InviewLandmarkIds,visited,2,candidate.globalKeyframeID - 1);
    getInviewPoint(InviewLandmarkIds,visited,2,candidate.globalKeyframeID + 1);
    set<int> InviewLandmarkIds_current;
    set<int> visited_current;
    // processedID.clear();
    getInviewPoint(InviewLandmarkIds_current,visited_current,2,currentGlobalKeyframeId);
    getInviewPoint(InviewLandmarkIds_current,visited_current,2,currentGlobalKeyframeId - 1);
    

    visualizeFrame(visited);
    visualizeFrame(visited_current);
    set<int> intersection;
    // getInviewPoint(InviewLandmarkIds,visited,2,10);
    vector<shared_ptr<Landmark>> landmarks;
    ROS_DEBUG_STREAM(" Total inview Id " << InviewLandmarkIds.size());
    for (auto id : InviewLandmarkIds){
        if (landmark_manager->landmarks.count(id) != 0)
            landmarks.push_back(landmark_manager->landmarks[id]);
            if (InviewLandmarkIds_current.count(id) != 0){
                intersection.insert(id);
            }
    }
    //auto landmarks = landmark_manager->getVisibleMapPoint(currentGlobalKeyframeId,stateTose3(states[currentGlobalKeyframeId]),processedID,ProjectedLocations);
    ROS_DEBUG_STREAM(" Total inview Point size " << landmarks.size());
    if (landmarks.size() ==0) return;
    //get descriptor
    cv::Mat ThreeDdescriptors;
    vector<cv::Point3f> point_3d;
    unordered_map<int,int> globalId_index_map;
    cnt = 0;
    for (auto ld:landmarks){
        ThreeDdescriptors.push_back(ld->descriptor);
        auto mapPointsInLast = camera_->world2camera(ld->pointGlobal,stateTose3(states[candidate.globalKeyframeID]));
        point_3d.push_back(eigenTocv(mapPointsInLast)); 
        globalId_index_map[cnt] = ld->landmarkId;
        cnt ++;

        loopClosurePoint.push_back({ld,1});
    }
    cout << "3d descriptors size " << point_3d.size() << endl;
    cout << "2d descriptors size " << unmatched2dPoint.size() << endl;
    if (ThreeDdescriptors.empty()){
        cout << "descriptor is empty 3d " << endl;
    }
    if (unmatched2dDescriptors.empty()){
        cout << "descriptor is empty 2d " << endl;
    }
    // bf match
    vector<cv::DMatch> matches;
    cv::BFMatcher      matcher(cv::NORM_L2, true);
    matcher.match(ThreeDdescriptors, unmatched2dDescriptors, matches);
    if (matches.size() == 0) return;
    
     // find the largest and smallest feature distances
    double min_dist = 100000, max_dist = 0;
    for (int i = 0; i < matches.size(); i++) {
        double dist = matches[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }
    vector<cv::DMatch>         pass_match;
    vector<cv::Point2f>        good2dPoint;
    vector<cv::Point2f>        goodProjection;
    vector<cv::Point3f>        good3dPoint;
    cout << "bf match size " << matches.size() << endl;
    for (int i = 0; i < matches.size(); i++) {
        if (  matches[i].distance <= 4 * min_dist ) {
            pass_match.push_back(matches[i]);
            good2dPoint.push_back(unmatched2dPoint[matches[i].trainIdx]);
            good3dPoint.push_back(point_3d[matches[i].queryIdx]);
            // goodProjection.push_back(ProjectedLocations[matches[i].queryIdx]);

        }
    }
   
    if (good3dPoint.size() < 4){
        return;
    }
    cout << "-------------------" << endl;
    cv::Mat inliers, ransacRVectorGuess;
    // Rodrigues(ransacRGuess, ransacRVectorGuess);

    cv::solvePnPRansac(good3dPoint,
                            good2dPoint,
                            camera_->K_cv(),
                            config_->distort,
                            ransacRGuess,
                            ransacTGuess,
                            true,
                            300,
                            5,
                            0.99,
                            inliers,
                            cv::SOLVEPNP_EPNP);
    curKey_globalId_map.clear();
    //Rodrigues(ransacRVectorGuess, ransacRGuess);
    // another map to avoid one to many match
    for (size_t i = 0; i < inliers.rows; ++i) {
        int mapPointId    = pass_match[inliers.at<int>(i, 0)].queryIdx;
        int keyPointId    = pass_match[inliers.at<int>(i, 0)].trainIdx;
        // cout << "distance " << pass_match[inliers.at<int>(i, 0)].distance << endl;
        if (keyPoint_index_map.count(keyPointId) == 0){
            ROS_ERROR_STREAM("keypoint map does not find");
        }
         if (globalId_index_map.count(mapPointId) == 0){
            ROS_ERROR_STREAM("globalId map does not find");
        }

        // temp 
        goodProjection = good2dPoint;
        if (curKey_globalId_map.count(keyPoint_index_map[keyPointId]) == 0)  {
            // auto projectedPoint = goodProjection[inliers.at<int>(i, 0)];
            // float distance = pow(projectedPoint.x -  good2dPoint[inliers.at<int>(i, 0)].x,2.0);
            // distance += pow(projectedPoint.y -  good2dPoint[inliers.at<int>(i, 0)].y,2.0);
            // distance = sqrt(distance);
            double distance = 0;
            if (distance < 100 ){
                loopClosurePoint[mapPointId].second = 2;
                if (visualizePointMatch(globalId_index_map[mapPointId],good2dPoint[inliers.at<int>(i, 0)],goodProjection[inliers.at<int>(i, 0)])){
                    curKey_globalId_map[keyPoint_index_map[keyPointId]] = globalId_index_map[mapPointId];
                }
            }
        }
       
}
    for (auto id: InviewLandmarkIds_current){
        if (landmark_manager->landmarks.count(id) != 0){
            loopClosurePoint.push_back({landmark_manager->landmarks[id],3});
        }
    }
    ROS_DEBUG_STREAM("Total Intersection "<<  intersection.size());
    for (auto id: intersection){
        if (landmark_manager->landmarks.count(id) != 0){
            loopClosurePoint.push_back({landmark_manager->landmarks[id],4});
        }
    }
    cout << "pnp inlier count " << inliers.rows << endl;
    cout <<"total keypoints " << currentKeypoints.size() << "total key points matched with last frame " << ransac_matches_id_map.size() << endl;
    cout << "total keypoints matched with other 3d points " << inliers.rows << endl;
    

}
void LoopClosingTool::getInviewPoint(set<int>& inViewLandmark,set<int>& visited,int level,int startFrame){
    
    if (level == 0){
        return;
    }
    if (visited.count(startFrame) != 0){ 
            return;
    }
    visited.insert(startFrame);
    ROS_DEBUG_STREAM(level << "Total connection of frame "<< startFrame << " is " << keyframes_[startFrame].connectedFrame.size());
    for (int Id : keyframes_[startFrame].globalIDs){
        if (processedID.count(Id) == 0){
            if (landmark_manager->landmarks[Id]->optimized)
                inViewLandmark.insert(Id);
        }
    }
    ROS_DEBUG_STREAM(inViewLandmark.size());
    for (auto connectedFrame:keyframes_[startFrame].connectedFrame){
        getInviewPoint(inViewLandmark,visited,level-1,connectedFrame);
      
    }
}
void LoopClosingTool::visualizeFrame(set<int> frames){
    for (auto frameID: frames){
        vector<cv::KeyPoint> newKeypoints;
        vector<cv::KeyPoint> detectedKeypoints;
        vector<cv::DMatch>   projectDetectMatch;
        if (keyframes_.count(frameID) != 0){
            int index = 0;
            for (auto landmarkID:keyframes_[frameID].globalIDs){
                if(landmark_manager->landmarks.count(landmarkID)){
                    if (landmark_manager->landmarks[landmarkID]->optimized){
                        cv::KeyPoint newKeypoint;
                        auto point3d = landmark_manager->landmarks[landmarkID]->pointGlobal;
                        auto projectedLocation = camera_->world2pixel(eigenTocv(point3d),stateTose3(states[frameID]));
                        newKeypoint.pt = projectedLocation;
                        newKeypoints.push_back(newKeypoint);
                        detectedKeypoints.push_back(keyframes_[frameID].keypoints[index]);

                        cv::DMatch newmatch;
                        newmatch.queryIdx = detectedKeypoints.size() -1;
                        newmatch.trainIdx = detectedKeypoints.size() -1;
                        projectDetectMatch.push_back(newmatch);
                    }
                }
                index++;
            }
        }
        cv::Mat projectionPic; 
        cv::Mat detectionPic; 
        cv::Mat matchPic;
        

        if (!newKeypoints.empty()){
            cv::drawKeypoints(keyframes_[frameID].img,newKeypoints,projectionPic,cv::Scalar::all(-1));
            cv::drawKeypoints(keyframes_[frameID].img,detectedKeypoints,detectionPic,cv::Scalar::all(-1));
            cv::imwrite("/home/bigby/ws/catkin_ws/Frame_id" + to_string(frameID) +".png",projectionPic);
            cv::imwrite("/home/bigby/ws/catkin_ws/Frame_id_detection" + to_string(frameID) +".png",detectionPic);
            cv::drawMatches(keyframes_[frameID].img, newKeypoints, keyframes_[frameID].img, detectedKeypoints, projectDetectMatch, matchPic, cv::Scalar(0, 0, 255), cv::Scalar::all(-1));
            cv::imwrite("/home/bigby/ws/catkin_ws/Frame_id_match" + to_string(frameID) +".png",matchPic);
        }
          
        	
    }
}
bool LoopClosingTool::visualizePointMatch(int landmarkID,cv::Point2f point,cv::Point2f projectedLocation){
    //get Landmark 
    auto projectedLandmark = landmark_manager->landmarks[landmarkID];
    cv::KeyPoint newKeypoint;
    newKeypoint.pt = point;
    cv::KeyPoint projectedKeypoint;
    projectedKeypoint.pt = projectedLocation;
    auto keypointIter = projectedLandmark->keypoints.end();
    keypointIter--;
    auto oldKeypoint = (keypointIter)->second;
    auto lastframeId = (keypointIter)->first;
    auto firstframeId = ( projectedLandmark->keypoints.begin())->first;
    auto oldImage = keyframes_[lastframeId].img;
    vector<cv::DMatch> match;
    cv::DMatch newmatch;
    newmatch.trainIdx = 0;
    newmatch.queryIdx = 0;
    match.push_back(newmatch);
    cv::Mat imMatches;
    cv::drawMatches(oldImage, {oldKeypoint}, currentImage, {newKeypoint,projectedKeypoint}, match, imMatches, cv::Scalar(0, 0, 255), cv::Scalar::all(-1));
    cout <<"current FrameId " << currentGlobalKeyframeId << " lastseen frame " << lastframeId << " of point " << landmarkID<< endl;
    
    cv::imwrite("/home/bigby/ws/catkin_ws/image(" +std::to_string(currentGlobalKeyframeId - lastframeId) +")(" + std::to_string(currentGlobalKeyframeId - firstframeId) + ")_" + std::to_string(keyframes_.size())+".jpg",imMatches);
    return true;
 
   


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