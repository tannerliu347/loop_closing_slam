#ifndef UTIL_H
#define UTIL_H
using namespace std;
#include <sophus/se3.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <opencv2/core/core.hpp>
#include <inekf_msgs/State.h>

/**
 * @brief bilinear interporation to compute the pixel intensity at (x, y).
 */
float interpolate(const float *img_ptr, float x, float y, int w, int h);
/**
 * @brief Conversion from se(3) to rotation/translation matrix.
 */
void se3ToRt(const Eigen::VectorXf &pose, Eigen::Matrix3f &rot, Eigen::Vector3f &t);
/**
 * @brief Conversion from rotation/translation matrix to se(3).
 */
void RtTose3(const Eigen::Matrix3f &rot, const Eigen::Vector3f &t, Eigen::VectorXf &pose);
cv::Point3f eigenTocv(Eigen::Vector3f p_e);

cv::Point2f eigenTocv(Eigen::Vector2f p_e);
Eigen::Vector3f cvToeigen(cv::Point3f p_c);
Eigen::Vector2f cvToeigen(cv::Point2f p_c);
Sophus::SE3f stateTose3( inekf_msgs::State state);
vector<cv::Point2f> keypointToPoints(vector<cv::KeyPoint> kps);

#endif // SIMPLE_DVO_UTIL_H
