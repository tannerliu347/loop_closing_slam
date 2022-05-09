#include "LoopClosingTool.hpp"
#include <algorithm>
LoopClosingTool::LoopClosingTool(DBoW3::Database* pDB):pDB_(pDB),
                                                    frameGap_(5), 
                                                    minScoreAccept_(0.015),
                                                    featureType_(1),
                                                    featureCount_(2000){   
        camera_mat= (cv::Mat_<double>(3, 3) << parameter.FX, 0., parameter.CX, 0., parameter.FY, parameter.CY, 0., 0., 1.);
        
    }

bool LoopClosingTool::detect_loop(vector<int>& matchingindex){
    //create a temporal current keyframe
    cv::Mat cur_desc = currentDescriptors;
    cv::Mat img = currentImage;
    DBoW3::QueryResults rets;
    int maxId = std::max(int(pDB_->size() - frameGap_),0);
    int top = parameter.top_match;
    pDB_->query(cur_desc, rets,top, maxId);
    bool loop_detected = false;
    // simple logic check to filter out unwanted
    if (rets.empty()) {
        pDB_->add(cur_desc);
        goodKeypoints.clear();
        goodKeypoints = currentKeypoints;
        generateKeyframe();
        return false;
    }
    //make sure closet frame have a good score
    if (rets.size() >= 1 && rets[0].Score > 0.02){
        for (int i = 1; i < rets.size() && i < top ; i ++ ){
            DBoW3::Result r = rets[i];
            IC(r.Score);
            if (r.Score < minScoreAccept_) {
        // pDB_->addImg(img);
        // //histKFs_.push_back(kf);
        // //std::cout << "added img\n";
        // return false;
            continue;
        }
        int inlier = ransac_featureMatching(keyframes_[r.Id]);
        eliminateOutliersPnP(keyframes_[r.Id]);
        inlier = ransac_matches.size();
        IC(inlier);
        //int inlier = 100;
        int inlierThresh = 30;

        if (inlier > inlierThresh){
            std::cout << "Cur frame: " << pDB_->size() << std::endl;
            int Curframe =  pDB_->size() - 1;
            loop_detected = true;
            matchingindex.push_back(keyframes_[r.Id].globalKeyframeID);
            genearteNewGlobalId(keyframes_[r.Id]);
        }
        good_matches.clear();
        ransac_matches.clear();
      }
    }else{
       loop_detected = false; 
    }
    pDB_->add(cur_desc);
    generateKeyframe();
    //keyframes.push_back(img);
    //min-index ?

    return loop_detected;keyframes_[r.Id]
    cv::Mat cur_descr,candidate_descr;
    std::vector<cv::KeyPoint> cur_keypoints, candidate_keypoints;
    cur_descr = currentDescriptors;
    candidate_descr = candidate.descriptors;
    cur_keypoints =currentKeypoints;
    candidate_keypoints = candidate.keypoints;
    cv::Mat curImg = currentImage;
    cv::Mat candidateImg = candidate.img;
    //create a matcher (Flann ) 
    
    if ( cur_descr.empty() ){
        IC("failure cur");
    }
          
    if ( candidate_descr.empty() ){
        IC("failure other");
    }
    // matcher.match( cur_descr, candidate_descr, matches );
    //mathcer brute force
    cv::BFMatcher matcher(cv::NORM_HAMMING,true);
    std::vector<cv::DMatch> matches;
    matcher.match( cur_descr, candidate_descr, matches );
    cv::Mat match_img;
    
    //find good match
    //if a distance is bigger then 2*min distance, we assume is false
    //to avoid extrme small mindistance we use 30 
    std::vector<cv::DMatch> normalMatches;
    // auto min_max = std::minmax_element(matches.begin(),matches.end(),
    // [](const cv::DMatch &m1,const cv::DMatch &m2){return m1.distance < m2.distance;});
    // double min_d = min_max.first->distance;
    // double max_d = min_max.second->distance;
    // for (int i =0; i < matches.size();i++){
    //     if(matches[i].distance <= std::max(3*min_d,30.0)){
    //         normalMatches.push_back(matches[i]);
    //     }
    // }
    // //find rearange keypoint
    normalMatches = matches;
    std::vector<cv::Point2f> curKps, candKps;
        for (int i = 0; i < normalMatches.size(); i++) {
            candKps.push_back( candidate_keypoints[normalMatches[i].trainIdx].pt);
            curKps.push_back(cur_keypoints[normalMatches[i].queryIdx].pt);
    }
  
    // Use ransac to further remove outlier
    cv::Mat H;
    H = cv::findHomography(cv::Mat(curKps),cv::Mat(candKps),cv::RANSAC,parameter.RansacThresh2d);
    //check if H is empty
    if(H.empty()){return 0;}
    cv::Mat curTransformed;
    cv::perspectiveTransform(cv::Mat(curKps),curTransformed,H);
    // save inlier
    for (int i =0; i < normalMatches.size(); i ++){
        if (cv::norm(candKps[i] - curTransformed.at<cv::Point2f>((int)i,0)) <= parameter.RansacThresh2d){
            good_matches.push_back(normalMatches[i]);
        }
    }
     for (int i = 0; i < good_matches.size(); i++) {
            goodKeypoints.push_back(cur_keypoints[good_matches[i].queryIdx]);
    }
    // if (good_matches.size() > 70){

    //     cv::drawMatches(curImg,cur_keypoints,candidateImg,candidate_keypoints,good_matches,match_img, 
    //     cv::Scalar::all(-1), cv::Scalar::all(-1),
    //     std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    //     cv::imshow("matched_result",match_img);
    // }
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
    #endif
            break;
        case 4:
            detector = cv::KAZE::create();
            break;
        case 5:
            detector = cv::AKAZE::create();
            break;
        }
    detector->detect(currentImage, currentKeypoints);
    detector->compute(currentImage, currentKeypoints, currentDescriptors);
}
void LoopClosingTool::create_feature(std::vector<cv::KeyPoint> Keypoints){
    currentKeypoints.clear();
    currentDescriptors.release();
    currentKeypoints = Keypoints;
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
    #endif
            break;
        case 4:
            detector = cv::KAZE::create();
            break;
        case 5:
            detector = cv::AKAZE::create();
            break;
        }
    detector->compute(currentImage,Keypoints, currentDescriptors);
}
void LoopClosingTool::assignNewFrame(const cv::Mat &img,const cv::Mat &depth,int gloablKeyframeId,std::vector<int> globalID){
    currentImage = img;
    currentDepth = depth;
    currentGlobalKeyframeId =  gloablKeyframeId;
    current_globalIDs = globalID;

}
void LoopClosingTool::generateKeyframe(){
    //calculate point 3d ]
    get3DfeaturePosition(point_3d, currentDepth, goodKeypoints);
    if (keyframes_.empty()){
        Keyframe kf = Keyframe(0,currentImage,currentDepth,currentKeypoints,goodKeypoints,currentDescriptors,currentGlobalKeyframeId);
        kf.insertGlobalID(current_globalIDs);
        kf.insertPoint3D(point_3d);
        keyframes_.push_back(kf);
    }else{
        Keyframe kf = Keyframe(keyframes_.back().frameID+1,currentImage,currentDepth,currentKeypoints,goodKeypoints,currentDescriptors,currentGlobalKeyframeId);
        kf.insertGlobalID(current_globalIDs);
        keyframes_.push_back(kf);
        kf.insertPoint3D(point_3d);
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

void LoopClosingTool::eliminateOutliersPnP(Keyframe& candidate){
    ransac_matches.clear();
    vector<cv::Point3f> candidate_3d; //3d point from candidate
    get3DfeaturePosition(candidate_3d, candidate.depth,candidate.good_keypoints);
    //candidate_3d = candidate.point_3d;
    get2DfeaturePosition(point_2d,goodKeypoints);
    // if(candidate_3d.size() == 0){
    //     throw "candidate_3d point empty";
    // }
    if (candidate_3d.size() >= 4) {
        cv::Mat inliers, ransacRVectorGuess;
        Rodrigues(ransacRGuess, ransacRVectorGuess);
        //convert point 3d 2d size to same size
        vector<cv::Point2f> point_2d_use;
        vector<cv::Point3f>point_3d_use;
        if (point_2d.size() >=  candidate_3d.size()){
            point_3d_use = candidate_3d;
            for(int i = 0 ; i <  candidate_3d.size() ;i ++){
                point_2d_use.push_back(point_2d[i]);
            }
        }else {
            point_2d_use = point_2d;
            for(int i = 0 ; i <  point_2d.size() ;i ++){
                point_3d_use.push_back(candidate_3d[i]);
            }
        }
        cv::solvePnPRansac(point_3d_use,
                           point_2d_use,
                           camera_mat,
                           distort,
                           ransacRVectorGuess,
                           ransacTGuess,
                           true,
                           parameter.ransacIterations,
                           parameter.ransacReprojectionError,
                           0.99,
                           inliers,
                           cv::SOLVEPNP_EPNP);

        ransac_matches.clear();
        ransac_matches_id_map.clear();
        for (size_t i = 0; i < inliers.rows; ++i) {
            ransac_matches.push_back(good_matches[inliers.at<int>(i, 0)]);
            ransac_matches_id_map.insert({good_matches[inliers.at<int>(i, 0)].queryIdx, good_matches[inliers.at<int>(i, 0)].trainIdx});
        }
    } else {
        ransac_matches = good_matches;
        for (size_t i = 0; i < ransac_matches.size(); ++i) {
            ransac_matches_id_map.insert({good_matches[i].queryIdx, good_matches[i].trainIdx});
        }
    }
    cv::Mat lastImage = candidate.img;
    std::vector<cv::KeyPoint> lastKeypoints = candidate.keypoints;
    cv::Mat imMatches;
    cv::drawMatches(currentImage,currentKeypoints, lastImage, lastKeypoints, ransac_matches, imMatches, cv::Scalar(0, 0, 255), cv::Scalar::all(-1));
    cv::imshow("matches_window", imMatches);
    cv::waitKey(1);
    cv::imwrite("result" +std::to_string(id) + ".png",imMatches );
    id++;
    // try {
        
    //     cv::drawMatches(currentImage, currentKeypoints, lastImage, lastKeypoints, ransac_matches, imMatches, cv::Scalar(0, 0, 255), cv::Scalar::all(-1));
    //     cv::imshow("matches_window", imMatches);
    //     //cv::waitKey(1);
    //     cv::waitKey(1);
    // } catch (...) {
    //     cout << "ERROR" << endl;
    // }
    cout << "match size: " << good_matches.size() << "," << ransac_matches.size() << endl;
}
std::vector<int> LoopClosingTool::genearteNewGlobalId(Keyframe& candidate){
    std::vector<int> candidate_globalId = candidate.globalIDs;
    //check ransac matches, find matched global id, created map
    std::unordered_map<int,int> matched_globalId; 
    std::vector<int> result_globalId = current_globalIDs;
    for (int i = 0; i < ransac_matches.size(); i ++){
        result_globalId[ransac_matches[i].queryIdx] = candidate_globalId[ransac_matches[i].trainIdx];
    }
    current_globalIDs = result_globalId;
}