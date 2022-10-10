#include "LoopClosingTool.hpp"
#include <algorithm>
#include <opencv2/xfeatures2d.hpp>
LoopClosingTool::LoopClosingTool(fbow::Vocabulary* pDB):pDB_(pDB){   
        camera_mat= (cv::Mat_<double>(3, 3) << parameter.FX, 0., parameter.CX, 0., parameter.FY, parameter.CY, 0., 0., 1.);
        lastLoopClosure_ = -1;
        currentGlobalKeyframeId = 0;
        landmarks_.reset(new Landmarks());
    }

bool LoopClosingTool::detect_loop(vector<Matchdata>& point_matches){
    
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
    bool loop_detected = find_connection(keyframes_.back(),candidate_id,point_match,states);
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
        if(keyframes_[i].descriptors.empty()){
            //ROS_ERROR_STREAM("size" << keyframes_.size());
            ROS_DEBUG_STREAM(keyframes_[i].globalKeyframeID << " " << i  << "empty old descriptor");
            continue;
        }
        bowvector_cur = pDB_->transform(cur_desc);
        fbow::fBow bowvector_old;
        bowvector_old = pDB_->transform(keyframes_[i].descriptors);
        double score = fbow::fBow::score(bowvector_cur,bowvector_old);
        pq.push( std::make_pair (i, score));
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
            if (candidate_score < minScoreAccept_) {
        // pDB_->addImg(img)
        // return false;
            continue;
            } 
            if (keyframes_[candidate_id].descriptors.empty()){
                continue;
            }
            //calculate relative pose
            // auto current_state = current_state;
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
            int inlier = ransac_featureMatching(frame,keyframes_[candidate_id]);
            eliminateOutliersPnP(frame,keyframes_[candidate_id],pose);
            searchByProjection(frame,keyframes_[candidate_id]);
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
    ROS_DEBUG_STREAM("original match size "<< matches.size());
    matcher.match( candidate_descr,cur_descr, matches );
    cv::Mat match_img;

    //find good match
    //if a distance is bigger then 2*min distance, we assume is false
    //to avoid extrme small mindistance we use 30 
    std::vector<cv::DMatch> normalMatches;
    normalMatches = matches;
    ROS_DEBUG_STREAM("Normal Match size " << normalMatches.size());
    std::vector<cv::Point2f> curKps, candKps;
        for (int i = 0; i < normalMatches.size(); i++) {
            candKps.push_back(candidate_keypoints[normalMatches[i].trainIdx].pt);
            curKps.push_back(cur_keypoints[normalMatches[i].queryIdx].pt);
    }
  
    // // Use ransac to further remove outlier
    // cv::Mat H;
    // H = cv::findHomography(cv::Mat(curKps),cv::Mat(candKps),cv::RANSAC,parameter.RansacThresh2d);
    // //check if H is empty
    // if(H.empty()){return 0;}
    // cv::Mat curTransformed;
    // cv::perspectiveTransform(cv::Mat(curKps),curTransformed,H);
    // // save inlier
    // for (int i =0; i < normalMatches.size(); i ++){
    //     if (cv::norm(candKps[i] - curTransformed.at<cv::Point2f>((int)i,0)) <= parameter.RansacThresh2d){
    //         good_matches.push_back(normalMatches[i]);
    //     }
    // }
    good_matches.clear();
    good_matches = matches;
    auto candidate_3dpoints = candidate.get3dPoint();
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
           detector = cv::ORB::create();
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
        landmarks_->updateDescriptor(current_globalIDs,currentDescriptors);
    
    
}
void LoopClosingTool::assignNewFrame(const cv::Mat &img,const cv::Mat &depth,int gloablKeyframeId){
    currentImage = img;
    currentDepth = depth;
    currentGlobalKeyframeId =  gloablKeyframeId;
    //currentGlobalKeyframeId++;
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
   
    Keyframe kf(currentGlobalKeyframeId,currentImage,currentDepth,currentKeypoints,currentDescriptors,landmarks_);
    kf.insertGlobalID(current_globalIDs);
    keyframes_.push_back(kf);
    
    goodKeypoints.clear();
    currentKeypoints.clear();
    
   
}
void LoopClosingTool::get2DfeaturePosition(vector<cv::Point2f> &point_2d, const vector<cv::KeyPoint> &good_kp2){
    point_2d.clear();
    for (size_t i = 0; i < good_kp2.size(); ++i) {
        point_2d.push_back(good_kp2[i].pt);
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
        vector<cv::Point3f>point_3d_use;
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
                           camera_mat,
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
        if (ransac_matches.size() > inlier_ ){
            cv::drawMatches(lastImage, lastKeypoints, current.img, current.keypoints, ransac_matches, imMatches, cv::Scalar(0, 0, 255), cv::Scalar::all(-1));
            //cv::imshow("matches_window", imMatches);
            cv::namedWindow("image", cv::WINDOW_AUTOSIZE);
            cv::imshow("image", imMatches);
            if(ransac_matches.size() > 2){
            // cv::imwrite("/home/bigby/ws/catkin_ws/result" + std::to_string(id)+ ".bmp",imMatches );
            }
            //cv::drawMatches(lastImage, lastKeypoints, currentImage, currentKeypoints, ransac_matches, imMatches, cv::Scalar(0, 0, 255), cv::Scalar::all(-1));
            //cv::imshow("matches_window", imMatches);
            //cv::waitKey(1);
            cv::waitKey(1);
            ROS_DEBUG_STREAM("Total match " << ransac_matches.size() );
        }
        
    } catch (...) {
       ROS_ERROR_STREAM("failed to plot");
    }
    cout << current.keypoints.size() << " loopclosure pnp match size: " << good_matches.size() << "," << ransac_matches.size() << endl;
    
   
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
bool LoopClosingTool::NormlocalFrameLandmakrPos(int globalId,int frameID,cv::Point3f& result){
    auto ptGlobal = landmarks_->points3d[globalId]->getGlobalPos();
    auto CurrentState =  states[frameID];
    Eigen::Vector3f    positionVector(CurrentState.position.x, CurrentState.position.y, CurrentState.position.z);
    Eigen::Quaternionf poseOrientation(CurrentState.orientation.w, CurrentState.orientation.x, CurrentState.orientation.y, CurrentState.orientation.z);
    auto rotationMarix = poseOrientation.toRotationMatrix();
    Eigen::Matrix4f TransWC; // Your Transformation Matrix
    TransWC.setIdentity();   // Set to Identity to make bottom row of Matrix 0,0,0,1
    TransWC.block<3,3>(0,0) = rotationMarix;
    TransWC.block<3,1>(0,3) = positionVector;

    auto TransCW = TransWC.inverse();
    Eigen::Vector4f ptNorm = TransWC*Eigen::Vector4f(ptGlobal.x,ptGlobal.y,ptGlobal.z,1);
    //normalization 
    if (ptNorm[2] < 0){
        return false;
    } 
    ptNorm = ptNorm/ptNorm[2];
    auto ptLocal = Eigen::Vector3f(ptNorm[0], ptNorm[1], ptNorm[2]); 
    auto K = Eigen::Matrix3f();
    K  << parameter.FX, 0., parameter.CX, 0., parameter.FY, parameter.CY, 0., 0., 1.;
    ptLocal = K* ptLocal;
    result = cv::Point3f( ptLocal[0], ptLocal[1], ptLocal[2]);
    ROS_DEBUG_STREAM("originalPoint " << ptGlobal.x << " " << ptGlobal.y << " " << ptGlobal.z);
    ROS_DEBUG_STREAM("projectedPoint " << ptLocal[0] << " " << ptLocal[1]  << " " << ptLocal[2]);

    if (result.x < 0 || result.y < 0){
        return false;
    } 
    return true;
}
void LoopClosingTool::searchByProjection(Keyframe& current,Keyframe& candidate){
    ROS_DEBUG_STREAM("start search by Projection");
    unordered_map<int,int> matchedPoint;
    std::vector<int> current_globalIDs = current.globalIDs;
    std::vector<int> candidate_globalIDs = candidate.globalIDs;
    std::vector<cv::DMatch> additionalMatch;
    for (int i = 0; i < ransac_matches.size(); i ++){
        matchedPoint[ current_globalIDs[ransac_matches[i].trainIdx]] = 1;
        matchedPoint[ candidate_globalIDs[ransac_matches[i].queryIdx]] = 1;
    }
    auto candidateldGlobal = landmarks_->get3dPoint(candidate_globalIDs);
    auto currentldGlobal = landmarks_->get3dPoint(current_globalIDs);
    float threshDistance = 10000;
    for (int i = 0; i < candidate_globalIDs.size(); i++){
        int id = candidate_globalIDs[i];

        if (matchedPoint.count(id) != 0){
            //if the landmark already matched to other landmark skip
            continue;
        }
        
        cv::Point3f ptLocal;
        if (!NormlocalFrameLandmakrPos(id,current.globalKeyframeID,ptLocal)){
            continue;
        }
        //search closest point
        float minDistance = 1e9;
        float minID = -1;
        for (int j = 0; j < current_globalIDs.size(); j --){
            auto currentKeypoint = currentldGlobal[j];
            if (matchedPoint.count(current_globalIDs[j]) != 0){
                continue;
                
                
                            }
            float distance = norm(landmarks_->getDescriptor(id),current.descriptors.row(j),cv::NORM_L2);
            ROS_DEBUG_STREAM("landmark distance " << distance);
            //distance = sqrt(distance);
            if (distance < minDistance){
                minDistance = distance;
                minID = j;
            }
        }
        // if (minDistance > threshDistance){
        //     continue;
        // }else{
            cv::DMatch newMatch;
            newMatch.queryIdx = i;
            newMatch.trainIdx = minID;
            additionalMatch.push_back(newMatch);
            matchedPoint[current_globalIDs[minID]] = 1;
            matchedPoint[id] = 1;
        //}

    }

    if (!additionalMatch.empty()){
        cv::Mat imMatches;
        cv::drawMatches(candidate.img, candidate.keypoints, current.img, current.keypoints, additionalMatch,imMatches, cv::Scalar(0, 0, 255), cv::Scalar::all(-1));
        //cv::imshow("matches_window", imMatches);
        cv::namedWindow("image", cv::WINDOW_AUTOSIZE);
        cv::imshow("image", imMatches);
        id++;
        cv::imwrite("/home/bigby/ws/catkin_ws/result" + std::to_string(id)+ ".bmp",imMatches );
        
        //cv::drawMatches(lastImage, lastKeypoints, currentImage, currentKeypoints, ransac_matches, imMatches, cv::Scalar(0, 0, 255), cv::Scalar::all(-1));
        //cv::imshow("matches_window", imMatches);
        //cv::waitKey(1);
        cv::waitKey(1);
        ROS_DEBUG_STREAM("Total new match " << additionalMatch.size() );
    }
    

}
    
