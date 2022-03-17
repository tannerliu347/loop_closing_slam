#include "CvoLoopClosing.hpp"
#include <iostream>

namespace cvo {
    CvoLoopClosing::CvoLoopClosing(DBoW3::Database* pDB) : 
                                                                                            pDB_(pDB), 
                                                                                            frameGap_(10), 
                                                                                            minScoreAccept_(0.02) {

    }
    // change key frame to MAT
    bool CvoLoopClosing::detect_loop(const cv::Mat& kf) {
        // DBoW check
        DBoW3::QueryResults rets;
        const cv::Mat& currImg = kf.img_;
        pDB_->queryImg(currImg, rets,1, pDB_->size() - frameGap_);
        // simple logic check to filter out unwanted
        if (rets.empty()) {
            pDB_->addImg(currImg);
            histKFs_.push_back(kf);
            std::cout << "No candidate\n";
            return false;
        }
        DBoW3::Result r = rets[0];
        if (r.Score < minScoreAccept_) {
            pDB_->addImg(currImg);
            histKFs_.push_back(kf);
            std::cout << "added img\n";
            return false;
        }
        std::cout << "Cur frame: " << pDB_->size() << std::endl;
        std::cout << r << std::endl;
        // find initial guess for transform
        // extract features and match
        cv::Ptr<cv::ORB> detector = cv::ORB::create();
        std::vector<cv::KeyPoint> keypointPrev, keypointCurr;
        const cv::Mat& prevImg = histKFs_[r.Id].img_;
        cv::Mat descrPrev, descrCurr;
        detector->detectAndCompute(prevImg, cv::noArray(), keypointPrev, descrPrev);
        detector->detectAndCompute(currImg, cv::noArray() ,keypointCurr, descrCurr);
        cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::knnMatch());
        std::vector<std::vector<cv::DMatch>> matches;
        matcher->knnMatch(descrPrev, descrCurr, matches, 2);
        float ratioThresh = 0.75f;
        std::vector<cv::DMatch> goodMatches;
        for (int i = 0; i < matches.size(); i++) {
            if (matches[i][0].distance < ratioThresh * matches[i][1].distance)
                goodMatches.push_back(matches[i][0]);
        }
        std::vector<cv::Point2f> prevKps, currKps;

        int rejectionThresh
        for (int i = 0; i < goodMatches.size(); i++) {
            prevKps.push_back(keypointPrev[goodMatches[i].trainIdx].pt);
            currKps.push_back(keypointCurr[goodMatches[i].queryIdx].pt);
        }
        cv::Mat H = cv::findHomography(prevKps, currKps, cv::RANSAC);
        cv::Mat K = calib_->intrinsic();

        return true;
    }


    
} //namespace cvo