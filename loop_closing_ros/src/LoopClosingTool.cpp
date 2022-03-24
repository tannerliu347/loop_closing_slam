#include "LoopClosingTool.hpp"
#include <algorithm>
LoopClosingTool::LoopClosingTool(DBoW3::Database* pDB):pDB_(pDB),
                                                    frameGap_(5), 
                                                    minScoreAccept_(0.000001),
                                                    featureType_(1),
                                                    featureCount_(2000){   
        camera_mat= (cv::Mat_<double>(3, 3) << parameter.FX, 0., parameter.CX, 0., parameter.FY, parameter.CY, 0., 0., 1.);
        
    }

bool LoopClosingTool::detect_loop(vector<int>& matchingindex){
    //create a temporal current keyframe
    Keyframe currentKeyframe = Keyframe(0,currentImage,currentDepth,currentKeypoints,currentDescriptors,currentGlobalKeyframeId);
    currentKeyframe.insertPoint3D(point_3d);

    cv::Mat cur_desc = currentDescriptors;
    cv::Mat img = currentImage;
    DBoW3::QueryResults rets;
    int maxId = std::max(int(pDB_->size() - frameGap_),0);
    int top = parameter.top_match;
    pDB_->query(cur_desc, rets,top, maxId);
    // simple logic check to filter out unwanted
    if (rets.empty()) {
        pDB_->add(cur_desc);
        generateKeyframe();
        return false;
    }
    for (int i = 0; i < rets.size() && i < top ; i ++ ){
        DBoW3::Result r = rets[i];
        // if (r.Score < minScoreAccept_) {
        // // pDB_->addImg(img);
        // // //histKFs_.push_back(kf);
        // // //std::cout << "added img\n";
        // // return false;
        //     continue;
        // }
        IC("here");
        int inlier = ransac_featureMatching(currentKeyframe,keyframes_[r.Id]);
        IC("here2");
        eliminateOutliersPnP(currentKeyframe,keyframes_[r.Id]);
        IC("here3");
        inlier = ransac_matches.size();
        IC(inlier);
        //int inlier = 100;
        int inlierThresh = 70;

        if (inlier > inlierThresh){
            std::cout << "Cur frame: " << pDB_->size() << std::endl;
            int Curframe =  pDB_->size() - 1;
            matchingindex.push_back(keyframes_[r.Id].globalKeyframeID);
        }
        
    }
    pDB_->add(cur_desc);
    generateKeyframe();
    //keyframes.push_back(img);
    return true;
}
int LoopClosingTool::ransac_featureMatching(Keyframe& current,Keyframe& candidate){
    //clear previous matches 
    good_matches.clear();

    cv::Mat cur_descr,candidate_descr;
    std::vector<cv::KeyPoint> cur_keypoints, candidate_keypoints;
    cur_descr = current.descriptors;
    candidate_descr = candidate.descriptors;
    cur_keypoints = current.keypoints;
    candidate_keypoints = candidate.keypoints;
    cv::Mat curImg = current.img;
    cv::Mat candidateImg = candidate.img;
    //create a matcher (Flann ) 
    if ( cur_descr.empty() ){
        IC("failure cur");
    }
          
    if ( candidate_descr.empty() ){
        IC("failure other");
    }
    
    // cv::FlannBasedMatcher matcher;
    // std::vector< cv::DMatch > matches;
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
    auto min_max = std::minmax_element(matches.begin(),matches.end(),
    [](const cv::DMatch &m1,const cv::DMatch &m2){return m1.distance < m2.distance;});
    double min_d = min_max.first->distance;
    double max_d = min_max.second->distance;

    for (int i =0; i < matches.size();i++){
        if(matches[i].distance <= std::max(2*min_d,30.0)){
            normalMatches.push_back(matches[i]);
        }
    }
    normalMatches = matches;
    //find rearange keypoint
    std::vector<cv::Point2f> curKps, candKps;
        for (int i = 0; i < normalMatches.size(); i++) {
            candKps.push_back( candidate_keypoints[normalMatches[i].trainIdx].pt);
            curKps.push_back(cur_keypoints[normalMatches[i].queryIdx].pt);
    }
    good_matches = normalMatches;
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
            good_matches.push_back(matches[i]);
        }
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
    detector->compute(currentImage, currentKeypoints, currentDescriptors);
}
void LoopClosingTool::assignNewFrame(const cv::Mat &img,const cv::Mat &depth,int gloablKeyframeId,std::vector<int> globalID){
    currentImage = img;
    currentDepth = depth;
    currentGlobalKeyframeId =  gloablKeyframeId;
    current_globalIDs = globalID;

}
void LoopClosingTool::generateKeyframe(){
    //calculate point 3d 
    //get3DfeaturePosition(point_3d, currentDepth, currentKeypoints);
    if (keyframes_.empty()){
        Keyframe kf = Keyframe(0,currentImage,currentDepth,currentKeypoints,currentDescriptors,currentGlobalKeyframeId);
        kf.insertGlobalID(current_globalIDs);
        kf.insertPoint3D(point_3d);
        keyframes_.push_back(kf);
    }else{
        Keyframe kf = Keyframe(keyframes_.back().frameID+1,currentImage,currentDepth,currentKeypoints,currentDescriptors,currentGlobalKeyframeId);
        kf.insertGlobalID(current_globalIDs);
        keyframes_.push_back(kf);
        kf.insertPoint3D(point_3d);
    }

   
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

void LoopClosingTool::eliminateOutliersPnP(Keyframe& current,Keyframe& candidate){
    vector<cv::Point3f> candidate_3d = candidate.point_3d; //3d point from candidate
    //get2DfeaturePosition(point_2d,currentKeypoints);
    IC(candidate_3d.size());
    IC(point_2d.size());
    IC(good_matches.size());
    if (candidate_3d.size() >= 4) {
        cv::Mat inliers, ransacRVectorGuess;
        Rodrigues(ransacRGuess, ransacRVectorGuess);
        cv::solvePnPRansac(candidate_3d,
                           point_2d,
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
    try {
        
        cv::drawMatches(currentImage, currentKeypoints, lastImage, lastKeypoints, ransac_matches, imMatches, cv::Scalar(0, 0, 255), cv::Scalar::all(-1));
        cv::imshow("matches_window", imMatches);
        //cv::waitKey(1);
        cv::waitKey(0);
    } catch (...) {
        cout << "ERROR" << endl;
    }
    cout << "match size: " << good_matches.size() << "," << ransac_matches.size() << endl;
}
