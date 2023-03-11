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
    
}
#endif /* CONFIG_H */
