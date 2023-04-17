
#ifndef CAMERA_H
#define CAMERA_H

#include <sophus/se3.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <opencv2/core/core.hpp>
#include "util.h"


class Camera{
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    double fx_;
    double fy_;
    double cx_;
    double cy_;
    Sophus::SE3f Camera_IMU_; //from IMU to Camer //TODO: use this transforamtion ? 
    Sophus::SE3f IMU_Camera_;

    Camera(double fx,double fy,double cx,double cy):fx_(fx),
                                                    fy_(fy),
                                                    cx_(cx),
                                                    cy_(cy)
                                                            {
        IMU_Camera_ = Camera_IMU_.inverse();
    }
    Camera(){};

    void updateParameter(double fx,double fy,double cx,double cy){
        fx_ = fx;
        fy_ = fy;
        cx_ = cx;
        cy_ = cy;
    }
    void updateExtrinsic(Sophus::SE3f IMU_Camera){
        IMU_Camera_ = IMU_Camera;
        Camera_IMU_ = IMU_Camera.inverse();
        // IMU_Camera_ = IMU_Camera.inverse();
        // Camera_IMU_ = IMU_Camera;
    }
    cv::Mat K_cv();

    Eigen::Matrix<float, 3, 3> K_eigen();

    Eigen::Vector3f world2camera(Eigen::Vector3f p_w,Sophus::SE3f T_w_i);

    Eigen::Vector3f camera2world(Eigen::Vector3f p_c,Sophus::SE3f T_w_i);
    Sophus::SE3f ImuTrajectoryToCamera(Sophus::SE3f T_w_i);
    Sophus::SE3f CameraTrajectoryToImu(Sophus::SE3f T_w_c);
  

    cv::Point2f camera2pixel(cv::Point3f p_c);

    cv::Point3f pixel2camera(cv::Point2f p_p, float depth);

    cv::Point2f world2pixel(cv::Point3f p_w, Sophus::SE3f T_w_i);

    cv::Point3f pixel2world(cv::Point2f p_p, Sophus::SE3f T_w_i, double depth);
};

#endif