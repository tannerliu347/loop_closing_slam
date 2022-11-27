import numpy as np
import matplotlib.pyplot as plt
import gtsam
from gtsam.utils import plot
# print(matplotlib.__version__)
from gtsam import Point3, Pose3, Rot3
from gtsam.symbol_shorthand import X
# gtsam.utils.plot.plot_trajectory(0,initial)

connectedId = {}
MatchingId_cur = []
MatchingId_old = []
Measurement = []
odometryPose = {}
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
        elif entry == "ConnectedID":
            id = int(processedLine[1])
            connectedId[id] = []
            for i in range(2,len(processedLine)):
                connectedId[id].append(int(processedLine[i]))
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

    f.close()
    loopPose = gtsam.Pose3(rq,t)
    return curId,oldId,loopPose

def loopPoseProp(mainLoopPose,currentEstimate,curId,oldId):
    loopPoseDict = {}
    maxIndex = curId
    T_wCurrent_corrected = mainLoopPose*currentEstimate.atPose3(oldId).inverse()
    for id in connectedId[curId]:
        # T_connected_current
        maxIndex = max(id,maxIndex)
        Tco_cu =  currentEstimate.atPose3(id).inverse() * currentEstimate.atPose3(curId)

        # Tword_connected
        Tw_co = (Tco_cu*T_wCurrent_corrected.inverse()).inverse()
        loopPoseDict[id] = Tw_co
    return loopPoseDict,maxIndex
def plotPose(imageFrame,estimates):
    titles = ["result","backend","inekf"]
    color = ['green','blue','red']
    index = 0
    fig = plt.figure(imageFrame)
    ax = fig.add_subplot(projection='3d')
    for estimate in estimates:
        print("plot " + titles[index])
        resultPoses = gtsam.utilities.extractPose3(estimate)
        t = resultPoses[:,9:12]
        ax.plot3D(t[:,0],t[:,1],t[:,2],color[index])
        ax.set_zlim3d(bottom=0.2, top=10)
        ax.set_title(titles[index])
        index+=1
    ax.legend(titles)
if __name__ == '__main__':
    curId,oldId,loopPose = readLoopData("/home/bigby/curly_slam/catkin_ws/loopClosureInfo.txt")
    oldId+=1
    print(curId,oldId)
    __, currentEstimate = gtsam.readG2o("/home/bigby/curly_slam/catkin_ws/test.g2o", True)
    currentEstimate = gtsam.Values(currentEstimate)
    # gtsam.utils.plot.plot_trajectory(1,currentEstimate)
    params = gtsam.GaussNewtonParams()
    priorNoise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(6)*0.1)
    poseNoise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(6)*0.1)
    loopNoise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(6)*1)
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()
    
    # add prior
    prevPose = currentEstimate.atPose3(1)
    initial.insert(1,currentEstimate.atPose3(1))
    initial.insert(2,currentEstimate.atPose3(2))
    initial.insert(3,currentEstimate.atPose3(3))
    graph.add(gtsam.PriorFactorPose3(1,currentEstimate.atPose3(1),priorNoise))
    graph.add(gtsam.PriorFactorPose3(2,currentEstimate.atPose3(2),priorNoise))
    graph.add(gtsam.PriorFactorPose3(3,currentEstimate.atPose3(3),priorNoise))
    loopPoseDict,maxId= loopPoseProp(loopPose,currentEstimate,curId,oldId)
    for id in range(4,maxId + 1):
        # check connected id 
        currentPose = odometryPose[id]
        for connected_id in connectedId[id]:
            connected_pose = odometryPose[connected_id]
            relativePose = connected_pose.inverse() * currentPose 
            graph.add(gtsam.BetweenFactorPose3(connected_id, id,relativePose, poseNoise))
        initial.insert(id,currentPose)
        
    params.setVerbosity(
        "Termination")  # this will show info about stopping conds
   
    # add loop closure
    # loopPose = currentEstimate.atPose3(oldId).inverse() * currentEstimate.atPose3(curId) 
    print(loopPose)
    graph.add(gtsam.BetweenFactorPose3(oldId, curId,gtsam.Pose3(), loopNoise))
    # add more loop closure constraint 

    for id,Tw_co in loopPoseDict.items():
        # T old_connected
        Told_co = currentEstimate.atPose3(oldId).inverse() * Tw_co
        # print(Told_co)
        loopclosureedge = gtsam.BetweenFactorPose3(oldId, id,Told_co, loopNoise)
        graph.add(loopclosureedge)
    optimizer_params = gtsam.LevenbergMarquardtParams()
    #     # optimizer_params.setVerbosityLM('SUMMARY')
    print('graph.size(): ', graph.size())
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, optimizer_params)
    # optimizer = gtsam.GaussNewtonOptimizer(graph, initial)
    result = optimizer.optimize()
    print('before optimizaitonerror(): ', graph.error(currentEstimate))
    print('after optimizaitonerror(): ', graph.error(result))
   
   
    # gtsam.utils.plot.plot_trajectory(0,result)
   
    plotPose(1,[result,currentEstimate, getGT()])
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