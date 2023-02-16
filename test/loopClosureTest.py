import numpy as np
import matplotlib.pyplot as plt
import gtsam
from gtsam.utils import plot
# print(matplotlib.__version__)
from gtsam import Point3, Pose3, Rot3
from gtsam.symbol_shorthand import X,L
# gtsam.utils.plot.plot_trajectory(0,initial)
from landmarkManager import LandmarkManager,Observation,Landmark,Frame
import inspect
import cv2
MatchingId_cur = []
MatchingId_old = []
Measurement = []
odometryPose = {}
landmarkManager = LandmarkManager()
def getGT():
    initial = gtsam.Values()
 
    for id,pose in odometryPose.items():
        initial.insert(id,pose)
    return initial
def readLoopData(path):
    curId = 0
    oldId = 0
    rq = None
    t = None
    f = open(path, "r")
    for line in f:
        processedLine = line.split()
        entry = processedLine[0]
        if (entry=="LoopClosureId"):
            curId = int(processedLine[1])
            oldId = int(processedLine[2])
        elif entry == "MatchingId":
            MatchingId_cur.append(int(processedLine[1]))
            MatchingId_old.append(int(processedLine[2]))
        elif entry == "Measurement":
            Measurement.append((float(processedLine[1]),float(processedLine[2])))
        elif entry == "Rotation":
            rq = Rot3(float(processedLine[1]),float(processedLine[2]),float(processedLine[3]) , float(processedLine[4]))
        elif entry == "Translation":
            t = np.array([float(processedLine[1]),float(processedLine[2]),float(processedLine[3])])
        elif entry == "odometryPose":
            id = int(processedLine[1])
            translation = np.array([float(processedLine[2]),float(processedLine[3]),float(processedLine[4])])
            rot = Rot3(float(processedLine[5]),float(processedLine[6]),float(processedLine[7]) , float(processedLine[8]))
            odometryPose[id] = gtsam.Pose3(rot,translation)
        elif entry == "ConnectedID":
            id = int(processedLine[1])
            if id not in landmarkManager.frames:
                landmarkManager.frames[id] = Frame(id)
            for i in range(2,len(processedLine)):
                landmarkManager.frames[id].connectedIds.append(int(processedLine[i]))
        elif entry == "Landmarks":
            landmarkId = int(processedLine[1])
            frameId = int(processedLine[2])
            observationData = (float(processedLine[3]),float(processedLine[4]))
            # create frame 
            if frameId not in landmarkManager.frames:
                landmarkManager.frames[frameId] = Frame(frameId)
            # create landmarks
            if landmarkId not in landmarkManager.landmarks:
                landmarkManager.landmarks[landmarkId] = Landmark(landmarkId)
            
            # create observations 
            observation = Observation(frameId,observationData)
            # save landmarks 
            landmarkManager.frames[frameId].observedLandmarks[landmarkId] = landmarkManager.landmarks[landmarkId] 
            landmarkManager.landmarks[landmarkId].observedFrames[frameId] = landmarkManager.frames[id]
                
            # store observation
            landmarkManager.landmarks[landmarkId].observations[frameId] = observation
        elif entry == "intrinsic":
            landmarkManager.cameraMatrix = gtsam.Cal3_S2(float(processedLine[1]), float(processedLine[2]), 0.0, float(processedLine[3]),float(processedLine[4]) )
        elif entry == "extrinsic":
            translation = np.array([float(processedLine[1]),float(processedLine[2]),float(processedLine[3])])
            rot = Rot3(float(processedLine[4]),float(processedLine[5]),float(processedLine[6]) , float(processedLine[7]))
            landmarkManager.cameraPose = gtsam.Pose3(rot,translation)
        elif entry == "LandmarksEstimate":
            landmarkId = int(processedLine[1])
            landmarkEstimation = gtsam.Point3(float(processedLine[2]),float(processedLine[3]),float(processedLine[4]))

            if landmarkId not in landmarkManager.landmarks:
                landmarkManager.landmarks[landmarkId] = Landmark(landmarkId)
            landmarkManager.landmarks[landmarkId].pointXYZ = landmarkEstimation
    f.close()
    loopPose = gtsam.Pose3(rq,t)
    return curId,oldId,loopPose
def windowOptimization(loopPose,curId,oldId,currentEstimate):
    poseNoise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(6)*1)
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()
    oldPose = currentEstimate.atPose3(oldId)
    newPose = currentEstimate.atPose3(curId)

    # add between factor 
    # add loop closure Pose 
    graph.add(gtsam.BetweenFactorPose3(oldId, curId,loopPose, poseNoise))
    initial.insert(oldId, oldPose)
    initial.insert(curId, newPose)
    
    # add connected Frame 
    # TODO: add all connected frame point ?
    loopPoseDict,maxId= loopPoseProp(loopPose,currentEstimate,curId,oldId)
    for id,Tw_co in loopPoseDict.items():
        # T old_connected
        Told_co = currentEstimate.atPose3(oldId).inverse() * Tw_co
        # print(Told_co)
        loopclosureedge = gtsam.BetweenFactorPose3(oldId, id,Told_co, loopNoise)
        graph.add(loopclosureedge)
        initial.insert(id, currentEstimate.atPose3(id))
    # add point 
    for  i in range(len(MatchingId_old)):
        pointId = MatchingId_old[i]
        pointId_new = MatchingId_cur[i]
        
        if pointId not in landmarkManager.landmarks:
            pass

        landmark = landmarkManager.landmarks[pointId]
        landmarkEstiamtion = landmarkManager.landmarks[pointId].pointXYZ
        pointNoise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(3) * 1)
        measurementNoise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(2) * 5)
        graph.add(gtsam.PriorFactorPoint3(
                            L(pointId), landmarkEstiamtion, pointNoise
                        ))
        initial.insert(L(pointId), landmarkEstiamtion)
        
        observation_old = landmark.observations[oldId].pointUV
        observation_old = gtsam.Point2(observation_old[0],observation_old[1])
        observation_new = Measurement[i]
        observation_new = gtsam.Point2(observation_new[0],observation_new[1])
        
        graph.add(gtsam.GenericProjectionFactorCal3_S2(
                        observation_old, measurementNoise, oldId,
                        L(pointId), landmarkManager.cameraMatrix, landmarkManager.cameraPose))

        graph.add(gtsam.GenericProjectionFactorCal3_S2(
                        observation_new, measurementNoise, curId,
                        L(pointId), landmarkManager.cameraMatrix, landmarkManager.cameraPose))

      
    optimizer_params = gtsam.LevenbergMarquardtParams()
    # optimizer_params.setVerbosityLM('SUMMARY')
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, optimizer_params)
    result = optimizer.optimize()
    oldPose = result.atPose3(oldId)
    newPose = result.atPose3(curId)

    
    return oldPose.inverse()*newPose
    


def pose3Tonp(pose):
    return np.array(pose.matrix())
   
def triangulation(pose1,psoe2,observation1,observation2):
    fx = landmarkManager.cameraMatrix.fx()
    fy = landmarkManager.cameraMatrix.fy()
    px = landmarkManager.cameraMatrix.px()
    py = landmarkManager.cameraMatrix.py()

    K = np.array([[fx, 0 ,px,0],[0,fy,py,0],[0,0,1,0]])
    T1= pose3Tonp(pose1)
    T2= pose3Tonp(psoe2)
    Point = cv2.triangulatePoints(K@T1, K@T2, np.array(observation1.pointUV), np.array(observation2.pointUV))
    Point = Point/Point[3]
    return Point[0:3]



def globalBA(loopPose,curId,oldId,currentEstimate):
    # gtsam.utils.plot.plot_trajectory(1,currentEstimate)
    pointNoise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(3) * 1)
    measurementNoise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(2) * 5)
    params = gtsam.GaussNewtonParams()
    priorNoise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(6)*0.1)
    poseNoise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(6)*0.1)
    loopNoise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(6)*0.1)
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()
    # gtsam pose Vs odometry pose
    gtsamPose = True
    
    print(loopPose)
    loopPose = windowOptimization(loopPose,curId,oldId,currentEstimate)
    
    # add prior
    prevPose = gtsam.Pose3()
    initial.insert(0,gtsam.Pose3())
    
    graph.add(gtsam.PriorFactorPose3(0,prevPose,priorNoise))
   
    loopPoseDict,maxId= loopPoseProp(loopPose,currentEstimate,curId,oldId)
    for id in range(1,maxId + 1):
        # check connected id 
        if not gtsamPose:
            currentPose = odometryPose[id]
        else:
            currentPose = currentEstimate.atPose3(id)
        if id in landmarkManager.frames:
            for connected_id in landmarkManager.frames[id].connectedIds:
                if (connected_id > maxId):
                    continue
                if not gtsamPose or connected_id == 0:
                    connected_pose = odometryPose[connected_id]
                else:
                    connected_pose = currentEstimate.atPose3(connected_id)
                relativePose = connected_pose.inverse() * currentPose 
                graph.add(gtsam.BetweenFactorPose3(connected_id, id,relativePose, poseNoise))
        
        initial.insert(id,currentEstimate.atPose3(id))

    # add landmark
    for pointId,landmark in landmarkManager.landmarks.items():
        if pointId not in landmarkManager.landmarks:
            continue
        if len(landmark.observedFrames) == 2:
            keys = list(landmark.observedFrames.keys())
            if (keys[0] == 0):
                keys[0] = 1
            if(keys[0] > maxId or keys[1] > maxId):
                continue
            pose1 = currentEstimate.atPose3(keys[0])
            pose2 = currentEstimate.atPose3(keys[1])
            observation1 = landmark.observations[keys[0]]
            observation2 = landmark.observations[keys[1]]
            triangulation(pose1,pose2,observation1,observation2)
        if len(landmark.observedFrames) < 2:
            continue
        landmarkEstiamtion = landmarkManager.landmarks[pointId].pointXYZ
        initial.insert(L(pointId), landmarkEstiamtion)
        
        for frameId, frame in landmark.observedFrames.items():
            if (frameId > maxId):
                continue
            oberservation =  landmark.observations[frameId].pointUV
            graph.add(gtsam.GenericProjectionFactorCal3_S2(
                       gtsam.Point2(oberservation[0],oberservation[1]) , measurementNoise, frameId,
                        L(pointId), landmarkManager.cameraMatrix, landmarkManager.cameraPose))

        
    params.setVerbosity(
        "Termination")  # this will show info about stopping conds
   

   
    optimizer_params = gtsam.LevenbergMarquardtParams()
    #     # optimizer_params.setVerbosityLM('SUMMARY')
    print('graph.size(): ', graph.size())
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, optimizer_params)
    # optimizer = gtsam.GaussNewtonOptimizer(graph, initial)
    result = optimizer.optimize()
    # print('before optimizaitonerror(): ', graph.error(currentEstimate))
    # print('after optimizaitonerror(): ', graph.error(result))
   
    return result
def loopPoseProp(mainLoopPose,currentEstimate,curId,oldId):
    loopPoseDict = {}
    maxIndex = curId
    T_wCurrent_corrected = mainLoopPose*currentEstimate.atPose3(oldId).inverse()
    if curId in landmarkManager.frames:
        for id in landmarkManager.frames[curId].connectedIds:
        # T_connected_current

            maxIndex = max(id,maxIndex)
        
            Tco_cu =  currentEstimate.atPose3(id).inverse() * currentEstimate.atPose3(curId)

            # Tword_connected
            Tw_co = (Tco_cu*T_wCurrent_corrected.inverse()).inverse()
            loopPoseDict[id] = Tw_co
    return loopPoseDict,maxIndex
def plotPose(imageFrame,estimates):
    titles = ["result","backend","inekf","globalBaResult"]
    color = ['green','blue','red','black']
    linewidth = [2,1,1,1]
    index = 0
    fig = plt.figure(imageFrame)
    ax = fig.add_subplot(projection='3d')
    for estimate in estimates:
        print("plot " + titles[index])
        resultPoses = gtsam.utilities.extractPose3(estimate)
        t = resultPoses[:,9:12]
        ax.plot3D(t[:,0],t[:,1],t[:,2],color[index], linewidth=linewidth[index])
        ax.set_zlim3d(bottom=0.2, top=10)
        ax.set_title(titles[index])
        index+=1
    ax.legend(titles)
if __name__ == '__main__':
    curId,oldId,loopPose = readLoopData("/home/bigby/curly_slam/catkin_ws/loopClosureInfo.txt")
    odometryPose[0] = gtsam.Pose3()
    print(curId,oldId)
    __, currentEstimate = gtsam.readG2o("/home/bigby/curly_slam/catkin_ws/test.g2o", True)
    pose3Tonp(currentEstimate.atPose3(0))
    currentEstimate = gtsam.Values(currentEstimate)
    # gtsam.utils.plot.plot_trajectory(1,currentEstimate)
    params = gtsam.GaussNewtonParams()
    priorNoise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(6)*1)
    poseNoise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(6)*1)
    loopNoise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(6)*0.01)
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()
    # gtsam pose Vs odometry pose
    gtsamPose = True

    # print(loopPose)
    # loopPose = windowOptimization(loopPose,curId,oldId,currentEstimate)
    
    # add prior
    prevPose = currentEstimate.atPose3(1)
    initial.insert(1,currentEstimate.atPose3(1))
    initial.insert(2,currentEstimate.atPose3(2))
    initial.insert(3,currentEstimate.atPose3(3))
    graph.add(gtsam.PriorFactorPose3(1,currentEstimate.atPose3(1),priorNoise))
    graph.add(gtsam.PriorFactorPose3(2,currentEstimate.atPose3(2),priorNoise))
    graph.add(gtsam.PriorFactorPose3(3,currentEstimate.atPose3(3),priorNoise))
    loopPoseDict,maxId= loopPoseProp(loopPose,currentEstimate,curId,oldId)
    print("maxId is " + str(maxId))
    
    for id in range(4,maxId + 1):
        # check connected id 
        if not gtsamPose:
            print("here")
            currentPose = odometryPose[id]
        else:
            currentPose = currentEstimate.atPose3(id)
        if id in landmarkManager.frames:
            for connected_id in landmarkManager.frames[id].connectedIds:
                if (connected_id > maxId):
                    continue
                if not gtsamPose:
                    connected_pose = odometryPose[connected_id]
                else:
                    connected_pose = currentEstimate.atPose3(connected_id)
                relativePose = connected_pose.inverse() * currentPose 
                graph.add(gtsam.BetweenFactorPose3(connected_id, id,relativePose, poseNoise))
        
        initial.insert(id,currentEstimate.atPose3(id))
        
    params.setVerbosity(
        "Termination")  # this will show info about stopping conds
   
    # add loop closure
    # loopPose = currentEstimate.atPose3(oldId).inverse() * currentEstimate.atPose3(curId) 
    
    graph.add(gtsam.BetweenFactorPose3(oldId, curId,loopPose, loopNoise))
    # add more loop closure constraint 

    for id,Tw_co in loopPoseDict.items():
        # T old_connected
        Told_co = currentEstimate.atPose3(oldId).inverse() * Tw_co
        # print(Told_co)
        loopclosureedge = gtsam.BetweenFactorPose3(oldId, id,Told_co, loopNoise)
        # graph.add(loopclosureedge)
    optimizer_params = gtsam.LevenbergMarquardtParams()
    #     # optimizer_params.setVerbosityLM('SUMMARY')
    print('graph.size(): ', graph.size())
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, optimizer_params)
    # optimizer = gtsam.GaussNewtonOptimizer(graph, initial)
    result = optimizer.optimize()
    # print('before optimizaitonerror(): ', graph.error(currentEstimate))
    # print('after optimizaitonerror(): ', graph.error(result))
   
    globalBAResult = globalBA(loopPose,curId,oldId,result)
    # gtsam.utils.plot.plot_trajectory(0,result)
   
    plotPose(1,[result,currentEstimate, getGT(),globalBAResult])
    #plotPose(1,[result,currentEstimate, getGT()])

    # plotPose(1,[result])

    # plt.ylim([-2,6])
    # plt.xlim([-6,6])
    # ax.set_zlim3d(bottom=0.2, top=10)


    # gtsam.utils.plot.plot_trajectory(1,initial)
    # plt.title("initial")
    # plot.plot_3d_points(0, result, linespec="g.")
    plt.show()


    # Optimize pose graph
    # print(t)
    # gtsam.Pose3(Rot3(),t)