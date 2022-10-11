#include "camera.h"
cv::Mat Camera::K_cv(){
    return cv::Mat1d(3, 3) << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1;
}

Eigen::Matrix<float, 3, 3> Camera::K_eigen(){
    Eigen::Matrix<float, 3, 3> k;
    k << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1;
    return k;
}

Eigen::Vector3f Camera::world2camera(Eigen::Vector3f p_w,Sophus::SE3f T_w_i){
    return Camera_IMU_*T_w_i.inverse() * p_w;
}

Eigen::Vector3f Camera::camera2world(Eigen::Vector3f p_c,Sophus::SE3f T_w_i){
    return T_w_i *IMU_Camera_* p_c;
}
Sophus::SE3f Camera::ImuTrajectoryToCamera(Sophus::SE3f T_w_i){
    return T_w_i *IMU_Camera_;
}
Sophus::SE3f Camera::CameraTrajectoryToImu(Sophus::SE3f T_w_c){
    return T_w_c *Camera_IMU_;
}
cv::Point2f Camera::camera2pixel(cv::Point3f p_c){
    float u =    fx_ * p_c.x / p_c.z + cx_;
    float v =    fy_ * p_c.x / p_c.z + cy_;
    return cv::Point2f(u,v);
}

cv::Point3f Camera::pixel2camera(cv::Point2f p_p, float depth){
    float x = (p_p.x - cx_) * depth / fx_;
    float y = (p_p.y - cy_) * depth / fy_;
    float z = depth;
    return cv::Point3f(x,y,z);
}
cv::Point2f Camera::world2pixel(cv::Point3f p_w, Sophus::SE3f T_w_i){
    return camera2pixel(eigenTocv(world2camera(cvToeigen(p_w), T_w_i)));
}

cv::Point3f Camera::pixel2world(cv::Point2f p_p, Sophus::SE3f T_w_i, double depth){
    return eigenTocv(camera2world(cvToeigen(pixel2camera(p_p, depth)), T_w_i));
}