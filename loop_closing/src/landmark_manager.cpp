#include "landmark_manager.h"




/*************************
**** Landmark Manager ****
**************************/

void LandmarkManagers::addKeyframe(Keyframes& keyframe) {
    if ( keyframe.keypoints.size() != keyframe.globalIDs.size() || keyframe.keypoints.size() )
    for (int i = 0 ; i < keyframe.keypoints.size(); i ++){
        int landmarkId = keyframe.globalIDs[i];
        shared_ptr<Landmarks> landmark;
        if (landmarks.count(landmarkId) == 0 ){
            landmark = make_shared<Landmarks>();
            landmarks[landmarkId] = landmark;
            landmark->landmarkId  = landmarkId;
        }
        landmark = landmarks[landmarkId];
        landmark->observedFrameIds.push_back(keyframe.globalKeyframeID);
        landmark->pointGlobal = cvToeigen(keyframe.point_3d[i]);
        //update descriptor
        landmark->updateDescriptor(keyframe.descriptors.row(i));
        landmark-> initiated = true;
        landmark->keypoints[keyframe.globalKeyframeID] = keyframe.keypoints[i];

    }
}
cv::Mat LandmarkManagers::getDescriptors(vector<unsigned int>& globalIDs){
    cv::Mat descriptors;
    for (int i = 0; i < globalIDs.size(); i ++){
        descriptors.push_back(landmarks[globalIDs[i]]->descriptor);
    }
    return descriptors;
}
bool LandmarkManagers::inView(unsigned int LandmarkID,Sophus::SE3f T_w_i,cv::Point2f& projectedLocation){
    auto t = T_w_i.translation();
    float distance = pow(landmarks[LandmarkID]->pointGlobal[0] - t[0],2.0);
    distance += pow(landmarks[LandmarkID]->pointGlobal[1] - t[1],2.0);
    distance += pow(landmarks[LandmarkID]->pointGlobal[2] - t[2],2.0);
    distance = sqrt(distance);
    // double angle = calculateViewAngle(T_w_i,LandmarkID);
    // cout << angle << endl;
    
    if (distance > 20 && distance < 1){
        return false;
    }
    auto ptInCamera =  camera->world2camera(landmarks[LandmarkID]->pointGlobal,T_w_i);
    if (ptInCamera[2] <0){
        return false;
    }
    projectedLocation = camera->camera2pixel(eigenTocv(ptInCamera));
    // cout << projectedLocation.x << " " << projectedLocation.y << endl;
    if (projectedLocation.x < 0 || projectedLocation.y  < 0 || projectedLocation.x  > config->imageWidth || projectedLocation.y > config->imageHeight ){
        return false;
    }
    

    return true;
}
 vector<shared_ptr<Landmarks>> LandmarkManagers::getVisibleMapPoint(int currentFrameId,Sophus::SE3f T_w_i,unordered_map<int,int>& processed,vector<cv::Point2f>& ProjectedLocations){
    // get all inview key point 
    // add point inview that haven'e been added yet.
    vector<shared_ptr<Landmarks>> outputPoints; 
    
    int cnt = 0;
    for (auto ld_pair: landmarks){
        auto landmark = ld_pair.second;
        if (processed.count(landmark->landmarkId) != 0){
            continue;
        }
        if (!landmark->optimized ){
            continue;
        }
        cv::Point2f location;
        if (inView(landmark->landmarkId,T_w_i,location)){
            cnt ++;
            outputPoints.push_back(landmark);
            ProjectedLocations.push_back(location);
        }

      
    }
    return outputPoints;
 };
void LandmarkManagers::updateLandmark(vector<unsigned int>& globalIds,vector<vector<double>>& points){
    currentProcessingGlobalId = globalIds;
    for (int i =0; i < globalIds.size(); i ++){
        int Id = globalIds[i];
        if (landmarks.count(Id) != 0){
            landmarks[Id]->pointGlobal[0] = points[i][0];
            landmarks[Id]->pointGlobal[1] = points[i][1];
            landmarks[Id]->pointGlobal[2] = points[i][2];
            landmarks[Id]->optimized = true;
        }
     

    }
}