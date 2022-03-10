#pragma once

#include <vector>
#include <DBoW3/DBoW3.h>
#include "KeyFrame.hpp"
#include <UnifiedCvo-0.1/cvo/CvoGPU.hpp>
#include <UnifiedCvo-0.1/utils/Calibration.hpp>



namespace cvo {
class CvoLoopClosing {
public:
    CvoLoopClosing(DBoW3::Database* pDB, cvo::CvoGPU* cvo, cvo::Calibration* calib);

    bool detect_loop(const KeyFrame& kf);

private:
    DBoW3::Database* pDB_;
    unsigned int frameGap_; // We consider frames within this range as too close
    float minScoreAccept_; // Disregard ones lower than this
    std::vector<KeyFrame> histKFs_;
    cvo::CvoGPU* cvo_;
    cvo::Calibration* calib_;
};
} //namespace cvo