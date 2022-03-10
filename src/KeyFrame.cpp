#include "KeyFrame.hpp"

namespace cvo {
KeyFrame::KeyFrame(const cv::Mat& l_img, const cvo::CvoPointCloud& cld) : img_(l_img), cld_(cld) {
    
}
}