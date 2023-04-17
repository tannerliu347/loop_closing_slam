#include "LoopClosureTestLoader.h"


LoopClosureTestLoader::LoopClosureTestLoader(std::string fronteEndFile, std::string backendFile){
       // check if file open successfully
        frameId   = 0;
        timestamp = 0;
        std::ifstream frontnendInfo;
        frontnendInfo.open(fronteEndFile);

        
        if (!frontnendInfo.is_open()) {
            LOG(FATAL) << "Cannot open frontend test file";
            exit(1);
        }
        int         currentframe = -1;
        std::string line;

        //read data from frontend file
        while (std::getline(frontnendInfo, line)) {
            std::istringstream iss(line);
            std::string token;
            iss >> token;
            
            if (token == "CAMERA_INTRINSIC") {
                iss >> camera.fx >> camera.cx >> camera.fy >> camera.cy;
            } else if (token == "CAMERA_POSE") {
                for (int i = 0; i < 7; i++) {
                    iss >> calibration.a[i];
                }
                // set translation first three are translation vector 
                gtsam::Point3 translation;
                for (int i = 0; i < 3; i++) {
                    translation(i) = calibration.a[i];
                }
                // set rotation quaternion xyzw
                gtsam::Rot3 quatRotation(calibration.a[6], calibration.a[3], calibration.a[4], calibration.a[5]);
            } else if (token == "FRAME") {
                currentframe++;
                FrameData frame;
                iss >> frame.id;
                frames.push_back(frame);
            } else if (token == "DATASET_SEQ") {
                iss >> frames[currentframe].dataset_seq;
            }
            else if (token == "ODOMETRY_POSE") {
                float a[7];
                for (int i = 0; i < 12; i++) {
                    iss >> a[i];
                }
                // frist three number is position
                Eigen::Vector3d position(a[0], a[1], a[2]);
                // last four number is quaternion
                Eigen::Vector4d quaternion(a[3], a[4], a[5], a[6]);
                frames[currentframe].odoemtryPose = gtsam::Pose3(gtsam::Rot3::Quaternion(quaternion(3), quaternion(0), quaternion(1), quaternion(2)), position);

            }else if (token == "FEATURE") {

                unsigned int id;
                double       u, v;
                double d, x, y, z;
                iss >> id >> u >> v >> d >> x >> y >> z;
                MeasuremnetData measurement;
                measurement.measurement = {u, v};
                measurement.point       = {x, y, z};
                measurement.point_id    = id;
                frames[currentframe].measurements[id] = measurement;
            }
        }

        frontnendInfo.close();
        currentframe = -1;      
        // read data from backend file
        std::ifstream backendInfo;
        backendInfo.open(backendFile);
        if (!backendInfo.is_open()) {
            LOG(FATAL) << "Cannot open backend test file";
            exit(1);
        }
        while (std::getline(backendInfo, line)) {
             std::istringstream iss(line);
            std::string token;
            iss >> token;
            
            if (token == "POSE") {
                for (int i = 0; i < 12; i++) {
                    iss >> calibration.a[i];
                }
                //set rotation. first 9 elements are rotation matrix
                gtsam::Matrix33 rotationMatrix;
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) {
                        rotationMatrix(i, j) = calibration.a[i * 3 + j];
                    }
                }
                //set translation. last 3 elements are translation vector
                gtsam::Point3 translation;
                for (int i = 0; i < 3; i++) {
                    translation(i) = calibration.a[i + 9];
                }
                auto rotation           = gtsam::Rot3(rotationMatrix);
                auto pose = gtsam::Pose3(rotation, translation);
                frames[currentframe].pose = pose;
            } else if (token == "FRAME") {
                currentframe++;
            } else if (token == "FEATURE") {

                unsigned int id;
                double       u, v;
                double d, x, y, z;
                iss >> id >> x >> y >> z >> u >> v;
                std::vector<double> point{x, y, z};
                frames[currentframe].measurements[id].point = point;
            }
        }
    LOG(INFO) << "Load test data successfully";
}