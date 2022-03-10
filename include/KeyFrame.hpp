#pragma once

#include <opencv2/core.hpp>
#include <UnifiedCvo-0.1/utils/CvoPointCloud.hpp>

namespace cvo {
class KeyFrame {
public:
    KeyFrame(const cv::Mat& l_img, const cvo::CvoPointCloud& cld);

    cv::Mat img_;
    cvo::CvoPointCloud cld_;
};

} //namespace cvo