#include <cmath>
#include <vector>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <memory>
#include <unordered_map>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <sophus/geometry.hpp>
#include <boost/thread.hpp>
#include <glog/logging.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <chrono>
#include <ctime>
    struct CameraData {
        float fx, cx, fy, cy;
    };

    struct CalibrationData {
        float a[12];
    };
    struct MeasuremnetData {
        unsigned int point_id;
        std::vector<double> measurement;
        std::vector<double> point;
    };
    struct FrameData {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        int id;
        int dataset_seq;
        gtsam::Pose3 odoemtryPose;
        gtsam::Pose3 pose;
        std::unordered_map<unsigned int,MeasuremnetData> measurements;
        std::vector<int> connections;
        Eigen::Vector4d quat;
        Eigen::Vector3d position;
        
    };

    class LoopClosureTestLoader{
    public:
        LoopClosureTestLoader(std::string fronteEndFile, std::string backendFile);
        CameraData camera;
        CalibrationData calibration;
        std::vector<FrameData> frames;
        int frameId;
        float timestamp;
        
    };
