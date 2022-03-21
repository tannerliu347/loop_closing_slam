#include "LoopClosingTool.hpp"
#include <algorithm>
LoopClosingTool::LoopClosingTool(DBoW3::Database* pDB,Eigen::MatrixXd* loopClosureMatch):pDB_(pDB),
                                                    loopClosureMatch_(loopClosureMatch),
                                                    frameGap_(30), 
                                                    minScoreAccept_(0.01),
                                                    featureType_(1),
                                                    featureCount_(3000){
    }

bool LoopClosingTool::detect_loop(const cv::Mat& img){
     // DBoW check
    DBoW3::QueryResults rets;
    int maxId = std::max(int(pDB_->size() - frameGap_),0);
    int top = 7;
    pDB_->queryImg(img, rets,top, maxId);
    // simple logic check to filter out unwanted
    if (rets.empty()) {
        pDB_->addImg(img);
        histKFs_.push_back(img);
        //std::cout << "No candidate\n";
        return false;
    }
    for (int i = 0; i < rets.size() && i < top ; i ++ ){
        DBoW3::Result r = rets[i];
        if (r.Score < minScoreAccept_) {
        // pDB_->addImg(img);
        // //histKFs_.push_back(kf);
        // //std::cout << "added img\n";
        // return false;
            continue;
        }
        int inlier = ransac_featureMatching(img,histKFs_[r.Id]);
        //int inlier = 100;
        int inlierThresh = 70;
        if (inlier > inlierThresh){
            std::cout << "Cur frame: " << pDB_->size() << std::endl;
            int Curframe =  pDB_->size() - 1;
            (*loopClosureMatch_)(Curframe,r.Id) = 1;
            (*loopClosureMatch_)(r.Id,Curframe) = 1;
        }
        
    }
    pDB_->addImg(img);
    histKFs_.push_back(img);
    return true;
}
bool LoopClosingTool::detect_loop(){
    // Keyframe* currentKeyframe;
    // if (keyframes_.empty()){
    //     IC("stop1");
    //     IC(currentDescriptors.rows);
    //     *currentKeyframe = Keyframe(0,currentImage,currentKeypoints,currentDescriptors);
    //     IC("past2");
    // }else{
    //     *currentKeyframe = Keyframe(keyframes_.back().frameID+1,currentImage,currentKeypoints,currentDescriptors);

    // }
    Keyframe currentKeyframe = Keyframe(0,currentImage,currentKeypoints,currentDescriptors);
    cv::Mat cur_desc = currentDescriptors;
    cv::Mat img = currentImage;
    DBoW3::QueryResults rets;
    int maxId = std::max(int(pDB_->size() - frameGap_),0);
    int top = 7;
    pDB_->query(cur_desc, rets,top, maxId);
    // simple logic check to filter out unwanted
    if (rets.empty()) {
        pDB_->add(cur_desc);
        generateKeyframe();
        return false;
    }
    for (int i = 0; i < rets.size() && i < top ; i ++ ){
        DBoW3::Result r = rets[i];
        if (r.Score < minScoreAccept_) {
        // pDB_->addImg(img);
        // //histKFs_.push_back(kf);
        // //std::cout << "added img\n";
        // return false;
            continue;
        }
        int inlier = ransac_featureMatching(currentKeyframe,keyframes_[r.Id]);
        //int inlier = 100;
        int inlierThresh = 70;
        if (inlier > inlierThresh){
            std::cout << "Cur frame: " << pDB_->size() << std::endl;
            int Curframe =  pDB_->size() - 1;
            (*loopClosureMatch_)(Curframe,r.Id) = 1;
            (*loopClosureMatch_)(r.Id,Curframe) = 1;
        }
        
    }
    pDB_->add(cur_desc);
    generateKeyframe();
    //keyframes.push_back(img);
    return true;
}
int LoopClosingTool::ransac_featureMatching(const cv::Mat& curImg, const cv::Mat& candidateImg){
    cv::Ptr<cv::ORB> detector = cv::ORB::create(3000);
    cv::Mat cur_descr,candidate_descr;
    std::vector<cv::KeyPoint> cur_keypoints, candidate_keypoints;
    // detect and compute feature
    detector->detectAndCompute(curImg, cv::noArray(), cur_keypoints,cur_descr);
    detector->detectAndCompute(candidateImg, cv::noArray() ,candidate_keypoints, candidate_descr);
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
    std::vector<cv::DMatch> goodMatches;
    auto min_max = std::minmax_element(matches.begin(),matches.end(),
    [](const cv::DMatch &m1,const cv::DMatch &m2){return m1.distance < m2.distance;});
    double min_d = min_max.first->distance;
    double max_d = min_max.second->distance;

    for (int i =0; i < matches.size();i++){
        if(matches[i].distance <= std::max(2*min_d,30.0)){
            goodMatches.push_back(matches[i]);
        }
    }
    goodMatches = matches;
    //find rearange keypoint
    std::vector<cv::Point2f> curKps, candKps;
        for (int i = 0; i < goodMatches.size(); i++) {
            candKps.push_back( candidate_keypoints[goodMatches[i].trainIdx].pt);
            curKps.push_back(cur_keypoints[goodMatches[i].queryIdx].pt);
    }
    
    // Use ransac to further remove outlier
    cv::Mat H;
    double ransacThresh = 20;
    H = cv::findHomography(cv::Mat(curKps),cv::Mat(candKps),cv::RANSAC,ransacThresh);
    //check if H is empty
    if(H.empty()){return 0;}
    cv::Mat curTransformed;
    cv::perspectiveTransform(cv::Mat(curKps),curTransformed,H);
    // save inlier
    std::vector<cv::DMatch> greatMatches;
    for (int i =0; i < goodMatches.size(); i ++){
        if (cv::norm(candKps[i] - curTransformed.at<cv::Point2f>((int)i,0)) <= ransacThresh){
            greatMatches.push_back(matches[i]);
        }
    }

    cv::drawMatches(curImg,cur_keypoints,candidateImg,candidate_keypoints,greatMatches,match_img, 
    cv::Scalar::all(-1), cv::Scalar::all(-1),
    std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imshow("matched",match_img);

    return greatMatches.size();
}
int LoopClosingTool::ransac_featureMatching(Keyframe& current,Keyframe& candidate){

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
    std::vector<cv::DMatch> goodMatches;
    auto min_max = std::minmax_element(matches.begin(),matches.end(),
    [](const cv::DMatch &m1,const cv::DMatch &m2){return m1.distance < m2.distance;});
    double min_d = min_max.first->distance;
    double max_d = min_max.second->distance;

    for (int i =0; i < matches.size();i++){
        if(matches[i].distance <= std::max(2*min_d,30.0)){
            goodMatches.push_back(matches[i]);
        }
    }
    goodMatches = matches;
    //find rearange keypoint
    std::vector<cv::Point2f> curKps, candKps;
        for (int i = 0; i < goodMatches.size(); i++) {
            candKps.push_back( candidate_keypoints[goodMatches[i].trainIdx].pt);
            curKps.push_back(cur_keypoints[goodMatches[i].queryIdx].pt);
    }
    
    // Use ransac to further remove outlier
    cv::Mat H;
    double ransacThresh = 20;
    H = cv::findHomography(cv::Mat(curKps),cv::Mat(candKps),cv::RANSAC,ransacThresh);
    //check if H is empty
    if(H.empty()){return 0;}
    cv::Mat curTransformed;
    cv::perspectiveTransform(cv::Mat(curKps),curTransformed,H);
    // save inlier
    std::vector<cv::DMatch> greatMatches;
    for (int i =0; i < goodMatches.size(); i ++){
        if (cv::norm(candKps[i] - curTransformed.at<cv::Point2f>((int)i,0)) <= ransacThresh){
            greatMatches.push_back(matches[i]);
        }
    }

    cv::drawMatches(curImg,cur_keypoints,candidateImg,candidate_keypoints,greatMatches,match_img, 
    cv::Scalar::all(-1), cv::Scalar::all(-1),
    std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imshow("matched",match_img);

    return greatMatches.size();
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
void LoopClosingTool::assignNewFrame(const cv::Mat &img){
    currentImage = img;
}
void LoopClosingTool::generateKeyframe(){
    if (keyframes_.empty()){

        keyframes_.push_back(Keyframe(0,currentImage,currentKeypoints,currentDescriptors));
    }else{
        keyframes_.push_back(Keyframe(keyframes_.back().frameID+1,currentImage,currentKeypoints,currentDescriptors));
    }
   
}
