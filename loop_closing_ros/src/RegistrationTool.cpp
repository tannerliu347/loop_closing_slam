#include "RegistrationTool.h"
bool RegistrationTool::Registration2d(RegistrationData2d& data){
    LOG(INFO) << "2d Match start " << endl;
    RobustMatcher robust_matcher;
    data.F = robust_matcher.match(data.img_candidate,data.img_cur,data.matches,data.points_candidate,data.points_cur,data.descriptors_candidate,data.descriptors_cur);
    if (data.F.rows != 3){
        return false;
    }
    data.inlierCount = data.matches.size();
    return true;
}
