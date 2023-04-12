#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <glog/logging.h>
#include <yaml-cpp/yaml.h>
using namespace std;

enum FeatureType { ORB, SURF, SIFT, KAZE, AKAZE };

struct LoopClosureParam {
    bool            cameraPoseInitialized = false;
    Eigen::Matrix3f cameraPoseRotation;
    Eigen::Vector3f cameraPoseTranslation;

    /* feature extraction & matching */
    FeatureType     featureType;
    int             featureNum;
    double          featureSampleNum;
    int             featureSampleGrid       = 5;
    double          featureMatchRange       = 3;
    string          featureExtractionMethod = "vanilla";
    Eigen::VectorXf keyframeSelectionWeight;
    double          ransacReprojectionError             = 8;
    int             ransacIterations                    = 500;
    bool            crossCheck                          = false;
    /* camera info */
    bool         cameraInfoInitialized = false;
    vector<double> cameraInfo;
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

    /*loop closuyre related */
    float min_score  = 0.15; // "Minimum number of score of a frame to be consider potential candidate"
    int inlier = 10; //"Minimum number of inlier"
    int top_match = 10; // "Top N frame from fbow database to check for potential loop closure candidate"
    int first_candidate = 10; // "Number of candidate to check for loop closure"
    int skip_frame = 0; // "Number of frame to skip before checking for loop closure"
    int near_frame = 0; //"number of frame to ignore when featching loop closure matching result"
    string vocab_path = "/home/robotics/Downloads/ORBvoc.txt"; // "Path to vocabulary file"


   
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    LoopClosureParam()
    : cameraPoseInitialized(false)
    , cameraInfoInitialized(false)
    , featureType(ORB)
    , featureNum(500)
    , featureSampleNum(200)
    , featureSampleGrid(5)
    , featureMatchRange(3)
    , featureExtractionMethod("vanilla")
    , ransacReprojectionError(8)
    , ransacIterations(500)
    , crossCheck(false)
    , depthScale(1000)
    , featureProjectionFrameNumber(5)
    , debugFlag(0)
    , timeIt(false)
    , showDepth(false)
    , showDepthX(0)
    , showDepthY(0)
    {
        keyframeSelectionWeight = Eigen::VectorXf::Ones(6);
    }
};
inline void read_LoopClosureParam_yaml(const char *filename, std::shared_ptr<LoopClosureParam> params) {
    YAML::Node fs = YAML::LoadFile(filename);
    std::cout << "open " << filename << std::endl;
    fs = fs["loop_closure"];
    if (fs["featureType"]) params->featureType = (FeatureType)fs["featureType"].as<int>();
    if (fs["featureNum"]) params->featureNum = fs["featureNum"].as<int>();
    if (fs["featureSampleNum"]) params->featureSampleNum = fs["featureSampleNum"].as<double>();
    if (fs["featureSampleGrid"]) params->featureSampleGrid = fs["featureSampleGrid"].as<int>();
    if (fs["featureMatchRange"]) params->featureMatchRange = fs["featureMatchRange"].as<double>();
    if (fs["featureExtractionMethod"]) params->featureExtractionMethod = fs["featureExtractionMethod"].as<string>();
    if (fs["ransacReprojectionError"]) params->ransacReprojectionError = fs["ransacReprojectionError"].as<double>();
    if (fs["ransacIterations"]) params->ransacIterations = fs["ransacIterations"].as<int>();
    if (fs["crossCheck"]) params->crossCheck = fs["crossCheck"].as<bool>();
    if (fs["featureProjectionFrameNumber"]) params->featureProjectionFrameNumber = fs["featureProjectionFrameNumber"].as<int>();
    if (fs["debugFlag"]) params->debugFlag = fs["debugFlag"].as<int>();
    if (fs["timeIt"]) params->timeIt = fs["timeIt"].as<bool>();
    if (fs["showDepth"]) params->showDepth = fs["showDepth"].as<bool>();
    if (fs["showDepthX"]) params->showDepthX = fs["showDepthX"].as<int>();
    if (fs["showDepthY"]) params->showDepthY = fs["showDepthY"].as<int>();
    if (fs["keyframeSelectionWeight"]) {
        params->keyframeSelectionWeight = Eigen::VectorXf::Zero(6);
        for (int i = 0; i < 6; i++) {
            params->keyframeSelectionWeight(i) = fs["keyframeSelectionWeight"][i].as<float>();
        }
    }
    if (fs["cameraInfo"]){
        params->cameraInfo = vector<double>(4);
        for (int i = 0; i < 4; i++) {
            params->cameraInfo[i] = fs["cameraInfo"][i].as<double>();
        }
    }
    if (fs["depthScale"]) params->depthScale = fs["depthScale"].as<double>();
    if (fs["imageWidth"]) params->imageWidth = fs["imageWidth"].as<unsigned int>();
    if (fs["imageHeight"]) params->imageHeight = fs["imageHeight"].as<unsigned int>();
    if (fs["cameraPoseRotation"]) {
        params->cameraPoseRotation = Eigen::Matrix3f::Zero();
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                params->cameraPoseRotation(i, j) = fs["cameraPoseRotation"][i][j].as<float>();
            }
        }
    }
    if (fs["cameraPoseTranslation"]) {
        params->cameraPoseTranslation = Eigen::Vector3f::Zero();
        for (int i = 0; i < 3; i++) {
            params->cameraPoseTranslation(i) = fs["cameraPoseTranslation"][i].as<float>();
        }
    }
    if (fs["min_score"]) params->min_score = fs["min_score"].as<float>();
    if (fs["inlier"]) params->inlier = fs["inlier"].as<int>();
    if (fs["top_match"]) params->top_match = fs["top_match"].as<int>();
    if (fs["first_candidate"]) params->first_candidate = fs["first_candidate"].as<int>();
    if (fs["skip_frame"]) params->skip_frame = fs["skip_frame"].as<int>();
    if (fs["near_frame"]) params->near_frame = fs["near_frame"].as<int>();
    if (fs["vocab_path"]) params->vocab_path = fs["vocab_path"].as<string>();
    
    
}
#endif /* CONFIG_H */
