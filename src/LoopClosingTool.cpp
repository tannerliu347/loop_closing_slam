#include "LoopClosingTool.hpp"
#include <algorithm>
LoopClosingTool::LoopClosingTool(DBoW3::Database* pDB,Eigen::MatrixXd* loopClosureMatch):pDB_(pDB),
                                                    loopClosureMatch_(loopClosureMatch),
                                                    frameGap_(40), 
                                                    minScoreAccept_(0.01){}

bool LoopClosingTool::detect_loop(const cv::Mat& img){
     // DBoW check
    DBoW3::QueryResults rets;
    int maxId = std::max(int(pDB_->size() - frameGap_),0);
    int top = 10;
    pDB_->queryImg(img, rets,top, maxId);
    // simple logic check to filter out unwanted
    if (rets.empty()) {
        pDB_->addImg(img);
        histKFs_.push_back(img);
        //std::cout << "No candidate\n";
        return false;
    }
    for (int i = 0; i < top; i ++ ){
        DBoW3::Result r = rets[i];
        if (r.Score < minScoreAccept_) {
        // pDB_->addImg(img);
        // //histKFs_.push_back(kf);
        // //std::cout << "added img\n";
        // return false;
            continue;
        }
        int inlier = ransac_featureMatching(img,histKFs_[r.Id]);
        int inlierThresh = 30;
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
    double ransacThresh = 30;
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
    //cv::drawMatches(curImg,curKps,candidateImg,candKps,goodMatches,match_img);
    cv::imshow("matched",match_img);

    return greatMatches.size();
}
// bool LoopClosingTool::detect_loop_test(const cv::Mat& img){
//      // DBoW check
//     DBoW3::QueryResults rets;
//     int maxId = std::max(int(pDB_->size() - frameGap_),0);
//     pDB_->queryImg(img, rets,1, maxId);
//     // simple logic check to filter out unwanted
//     if (rets.empty()) {
//         pDB_->addImg(img);
//         //histKFs_.push_back(kf);
//         //std::cout << "No candidate\n";
//         return false;
//     }
//     DBoW3::Result r = rets[0];
//     if (r.Score < minScoreAccept_) {
//         pDB_->addImg(img);
//         //histKFs_.push_back(kf);
//         //std::cout << "added img\n";
//         return false;
//     }
//     //std::cout << "Cur frame: " << pDB_->size() << std::endl;
//     int Curframe =  pDB_->size() - 1;
//     pDB_->addImg(img);
    
//     // //2d Ransac
//     // cv::Ptr<cv::ORB> detector = cv::ORB::create();
//     // std::vector<cv::KeyPoint> keypointPrev, keypointCurr;
//     // const cv::Mat& prevImg = histKFs_[r.Id].img_;
//     // cv::Mat descrPrev, descrCurr;
//     // detector->detectAndCompute(prevImg, cv::noArray(), keypointPrev, descrPrev);
//     // detector->detectAndCompute(currImg, cv::noArray() ,keypointCurr, descrCurr);
//     // cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::knnMatch());
//     // std::vector<std::vector<cv::DMatch>> matches;
//     // matcher->knnMatch(descrPrev, descrCurr, matches, 2);
//     // float ratioThresh = 0.75f;
//     // std::vector<cv::DMatch> goodMatches;
//     // for (int i = 0; i < matches.size(); i++) {
//     //     if (matches[i][0].distance < ratioThresh * matches[i][1].distance)
//     //         goodMatches.push_back(matches[i][0]);
//     // }
//     // std::vector<cv::Point2f> prevKps, currKps;
//     // for (int i = 0; i < goodMatches.size(); i++) {
//     //     prevKps.push_back(keypointPrev[goodMatches[i].queryIdx].pt);
//     //     currKps.push_back(keypointCurr[goodMatches[i].queryIdx].pt);
//     // }
//     // cv::Mat H = cv::findHomography(prevKps, currKps, cv::RANSAC);
//     // cv::Mat K = calib_->intrinsic();

//     (*loopClosureMatch_)(Curframe,r.Id) = 1;
//     //(*loopClosureMatch_)(r.Id,Curframe) = 1;
    
//     pDB_->addImg(img);


//     return true;
// }

