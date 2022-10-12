#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
using namespace std;

enum FeatureType { ORB, SURF, SIFT, KAZE, AKAZE };

struct Config {
    /* topic */
    string colorTopic;
    string depthTopic;
    string stateTopic;
    string cameraInfoTopic;

    // sensor poses
    std::string     worldFrame;
    std::string     bodyFrame;
    std::string     cameraFrame;
    bool            cameraPoseInitialized = false;
    Eigen::Matrix3f cameraPoseRotation;
    Eigen::Vector3f cameraPoseTranslation;

    /* feature extraction & matching */
    FeatureType     featureType             = ORB;
    int             featureNum              = 500;
    double          featureSampleNum        = 200;
    int             featureSampleGrid       = 5;
    double          featureMatchRange       = 3;
    string          featureExtractionMethod = "vanilla";
    Eigen::VectorXf keyframeSelectionWeight;
    double          ransacReprojectionError             = 8;
    int             ransacIterations                    = 500;
    bool            crossCheck                          = false;
    /* camera info */
    bool         cameraInfoInitialized = false;
    cv::Mat      cameraMatrix, distort;
    double       depthScale = 1000;
    unsigned int imageWidth;
    unsigned int imageHeight;

    /* feature projection */
    int featureProjectionFrameNumber = 5;

    /* debug */
    int  debugFlag  = 0;
    bool timeIt     = false;
    bool showDepth  = false;
    int  showDepthX = 0;
    int  showDepthY = 0;

    Config() {
        keyframeSelectionWeight = Eigen::VectorXf::Ones(6);
    }
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

#endif /* CONFIG_H */
