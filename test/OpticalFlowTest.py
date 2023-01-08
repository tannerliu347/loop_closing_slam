import numpy as np
import glob 
import cv2
import matplotlib.pyplot as plt
from landmarkManager import LandmarkManager,Landmark,Observation,Frame
Path = "/home/bigby/Downloads/tartanair_tools-master/seasidetown/Easy/seasidetown/seasidetown/Easy/P002/"

def quaternion_to_rotation_matrix(Q):
    # from https://automaticaddison.com/how-to-convert-a-quaternion-to-a-rotation-matrix/
    # Q: w x y z
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix
class flowParser():
    def __init__(self,flowPath):
        # Load flow mask
        fx = 320
        fy = 320 
        px = 320
        py = 240
        self.K = np.array([[fx, 0 ,px],[0,fy,py],[0,0,1]])

        self.imageH = 480
        self.imageW = 640
        self.featureNum = 100
        flowmaskPaths =  glob.glob(flowPath + 'flow/*mask.npy')
        self.flowmaskMap= {}
        for path in flowmaskPaths :
            filename = path.split('/')[-1]
            filename = filename.split('.')[0]
            flowIndex = int(filename.split('_')[0]) 
            self.flowmaskMap[flowIndex] = np.load(path)

        
        # Load flow
        self.flowMap= {}
        flowPaths =  glob.glob(flowPath + 'flow/*flow.npy')
        for path in flowPaths :
            filename = path.split('/')[-1]
            filename = filename.split('.')[0]
            flowIndex = int(filename.split('_')[0]) 
            self.flowMap[flowIndex] = np.load(path)

        
        # Load image 
        self.imageMap= {}
        imagePaths =  glob.glob(flowPath + 'image_left/*.png')
        for path in imagePaths :
            filename = path.split('/')[-1]
            filename = filename.split('.')[0]
            imageIndex = int(filename.split('_')[0]) 
            
            self.imageMap[imageIndex] = cv2.imread(path)
        
        # Load Pose 
        self.poses = {}
       
        with open(flowPath + "pose_left.txt", 'r') as file:
            lines = file.readlines()
            count = 0
            for line in lines:
                data = [float(x) for x in line[:-1].split(' ')]
                rot = quaternion_to_rotation_matrix(data[6:7]+data[3:6]).reshape(9).tolist()
                t = np.array([data[0],data[1],data[2]])
                self.poses[count] = (rot,t)
                count += 1 
        # Load Depth
        self.depthMap = {}
        
        depthPaths =  glob.glob(flowPath + 'depth_left/*.npy')
        
        for path in depthPaths :
            filename = path.split('/')[-1]
            filename = filename.split('.')[0]
            depthIndex = int(filename.split('_')[0]) 
            depthData = np.load(path)
            self.depthMap[depthIndex] = depthData



        # Frontend processing 
        self.currentKps = []
        self.nextKps = []
        self.matches = []
        self.curGlobalIDs= []
        self.matches_id_map = {}
        
        self.frameID = 0
        self.index = 0

        self.landmarkId = 0
        self.image  = self.imageMap[0]
        self.flowMask = self.flowmaskMap[0]
        self.flow = self.flowMap[0]
        self.frameGap = 1
        self.landmarkManager = LandmarkManager()

        # Todo: multi frame flow track
    def triangulation(self,pose1,pose2,observation1,observation2):
        fx = 320
        fy = 320 
        px = 320
        py = 240
        K = np.array([[fx, 0 ,px,0],[0,fy,py,0],[0,0,1,0]])

        t1 = pose1[1]
        rot1 = np.array(pose1[0]).reshape(3,3)
        t2 = pose2[1]
        rot2 = np.array(pose2[0]).reshape(3,3)
        T1 = np.zeros((4,4))
        T1[0:3,0:3] = rot1
        T1[0:3,3] = t1.T
        T1[3,3] = 1
   
        T2 = np.zeros((4,4))
        T2[0:3,0:3] = rot2
        T2[0:3,3] = t2.T
        T2[3,3] = 1
        
        Point = cv2.triangulatePoints(K@T1, K@T2, np.array(observation1.pointUV), np.array(observation2.pointUV))
        Point = Point/Point[3]
      
        return Point[0:3]
    def create_Keyframe(self):
        # go through current match 
        newGlobalId =[]
        for i in range(len(self.nextKps)):
            # get index to curKps
            cur_index = self.matches_id_map[i]
            globalId = 0
            if self.frameID not in self.landmarkManager.frames:
                newframe = Frame(self.frameID)
                self.landmarkManager.frames[self.frameID] = newframe
            if self.frameID + 1 not in self.landmarkManager.frames:
                newframe = Frame(self.frameID + 1)
                self.landmarkManager.frames[self.frameID + 1] = newframe
            if (cur_index >= len(self.curGlobalIDs)):
                globalId = self.landmarkId
                self.landmarkId+=1
                # new Point save both cur and next
                if globalId not in self.landmarkManager.landmarks:
                    newLandmark = Landmark(globalId)
                    depth_cur = self.depthMap[self.frameID][int(self.currentKps[cur_index].pt[1]),int(self.currentKps[cur_index].pt[0])]
                    depth_Ne = self.depthMap[self.index + self.frameGap][int(self.nextKps[i].pt[1]),int(self.nextKps[i].pt[0])]
                  
                    newObservationCur = Observation(self.frameID,self.currentKps[cur_index].pt,depth_cur)
                    newObservationNe = Observation(self.frameID + 1,self.nextKps[i].pt,depth_Ne)
                    newLandmark.observations[self.frameID] = newObservationCur
                    newLandmark.observations[self.frameID + 1] = newObservationNe
                    # Triangulation
                    poseCur = self.poses[self.frameID]
                    poseNe = self.poses[self.index + self.frameGap]
                    point = self.triangulation(poseCur,poseNe,newObservationCur,newObservationNe)
                    newLandmark.pointXYZ = point
                    self.landmarkManager.landmarks[globalId]= newLandmark
                    self.landmarkManager.frames[self.frameID + 1].observedLandmarks[globalId] = self.landmarkManager.landmarks[globalId]
                    self.landmarkManager.frames[self.frameID].observedLandmarks[globalId] = self.landmarkManager.landmarks[globalId]
            else:
                globalId = self.curGlobalIDs[cur_index]
                # old Point only save next 
                depth_Ne = self.depthMap[self.index + self.frameGap][int(self.nextKps[i].pt[1]),int(self.nextKps[i].pt[0])]
                newObservationNe = Observation(self.frameID + 1,self.nextKps[i].pt,depth_Ne)
                self.landmarkManager.landmarks[globalId].observations[self.frameID + 1] = newObservationNe
                self.landmarkManager.frames[self.frameID + 1].observedLandmarks[globalId] = self.landmarkManager.landmarks[globalId]
                
            

            newGlobalId.append(globalId)

        self.curGlobalIDs = newGlobalId
        


        # create new landmark or create 
    def mainThread(self):
        advance = True
        while advance:
            print("Processing frame " + str(self.frameID))
            print(self.featureNum - len(self.currentKps))
            if (self.featureNum - len(self.currentKps) > 0):
                self.detectNewLandmark(self.featureNum - len(self.currentKps))

            # Compute flow to next point 
            self.getFlow()
        
            self.create_Keyframe()
            # calcualte global point 
            self.currentKps = self.nextKps

            if (not advance):
                break
            self.frameID+=1
            self.index+=self.frameGap
            
            
            # Update image, flow, and flow mask
            try:
                self.image  = self.imageMap[self.frameID]
                self.flowMask = self.flowmaskMap[self.frameID]
                self.flow = self.flowMap[self.frameID]
            except:
                print("all frame parsed")
                self.saveFrontendResult(self.frameID)
                self.windowTest(206,113)

                return
            

        # update pointID and calculate triangulation Point 

        # 
    def saveFrontendResult(self,lastFrameId):
        f = open("groundtruthflowFrontend.txt",'w')
        f.write("FORMAT 1\n")
        f.write("CAMERA 320 320 320 240\n")
        f.write("CALIBRATION 0 0 1 1 0 0 0 1 0 0 0 0\n")
        
        for frameID in range(lastFrameId):
            f.write("FRAME " + str(frameID) +"\n")
            f.write("DATASET_SEQ " + str(frameID) +"\n")
            frame = self.landmarkManager.frames[frameID]
            for LandmarkId,ld in frame.observedLandmarks.items():
                f.write("FEATURE " + str(LandmarkId) + " ")
                observation = ld.observations[frameID]
                pointXYZ = ld.pointXYZ
                f.write(str(observation.pointUV[0]) + " " + str(observation.pointUV[1])+ " " +  str(observation.depth) + " ")
               
                f.write(str(pointXYZ[0,0]) + " " + str(pointXYZ[1,0]) + " " + str(pointXYZ[2,0]) + "\n" )
            # write camPose 
            pose = self.poses[frameID]
            rot = pose[0]
            t = pose[1]
            f.write("pose.rot\n")
            f.write(str(rot[0]) + " " + str(rot[1]) + " " + str(rot[2]) + "\n")
            f.write(str(rot[3]) + " " + str(rot[4]) + " " + str(rot[5]) + "\n")
            f.write(str(rot[6]) + " " + str(rot[7]) + " " + str(rot[8]) + "\n")
            f.write("pose.pos\n")
            f.write(str(t[0]) + " " + str(t[1]) + " " + str(t[2]) + "\n")
            f.write("cam.rot\n")
            f.write("0 0 1\n")
            f.write("1 0 0\n")
            f.write("0 1 0\n")
            f.write("cam.pos\n")
            f.write("0\n")
            f.write("0\n")
            f.write("0\n")


        
        # camera = ["CAMERA",320,320,320,240]
        # calibration = 
        f.close()

    def detectNewLandmark(self,featureCount):
        sift = cv2.SIFT_create(featureCount)
        additionalKps, des = sift.detectAndCompute(self.image, None)
        self.currentKps = self.currentKps + additionalKps
        # TODO: Remove redundant points

    def getFlow(self):
        self.matches_id_map = {} 
        self.matches = []
        self.nextKps = []
        # detect landmark 
        if self.frameID + 1 not in self.imageMap or self.frameID not in self.imageMap:
            return False
        image = self.image
        flowMask = self.flowMask
        flow = self.flow
        kp_next = []
        kp_cur = []
        index_cur = 0
        index_next = 0
        for kp in self.currentKps:
            u = int(kp.pt[0])
            v = int(kp.pt[1])
           
            if (flowMask[v,u] < 0 ):
                continue
            else:
                du = flow[v,u,0]
                dv = flow[v,u,1]
                u_next = kp.pt[0] + du
                v_next = kp.pt[1] + dv
                if (v_next < 0 or u_next < 0):
                    continue
                if (v_next > self.imageH  or u_next > self.imageW):
                    continue
                # else if (u )
                # u_next = du
                # v_next = dv 
                keypoint = cv2.KeyPoint()
                keypoint.pt = (u_next,v_next)
                kp_next.append(keypoint)
                kp_cur.append(kp)
                match = cv2.DMatch()
                match.queryIdx = index_cur
                match.trainIdx = index_next
                self.matches_id_map[index_next] = index_cur
                index_next += 1
                self.matches .append(match)
            index_cur+=1
        self.nextKps  = kp_next
        print("Current Total kp ", str(len(self.currentKps)))
        # Remove untrackablkey point 
        image_next = self.imageMap[self.frameID + 1]
        matched_image = cv2.drawMatches(image, kp_cur, image_next, kp_next, self.matches , None, flags=2)
        cv2.imshow('SIFT', matched_image)
        # cv2.waitKey(0)
        return True

    def windowTest(self,curId,oldId):
        oldPoints = []
        old3dPointsWorld = []
        old3dPointsLocal = []

        
        curPoints = []
        cur3dPointsWorld = []
        cur3dPointsLocal = []

        def pointWorldToCam(pose,pointWorld):
            T1 = np.zeros((4,4))
            T1[0:3,0:3] = np.array(pose[0]).reshape(3,3)
            T1[0:3,3] = pose[1].T
            T1[3,3] = 1
            pointWorld_h = np.zeros((4,1))
            pointWorld_h[0:3,:] = pointWorld
            pointLocal_h = np.linalg.inv(T1) @pointWorld_h
            pointLocal_h = pointLocal_h/pointLocal_h[3,0]
            return pointLocal_h[0:3,:]
            
        # get keypoints from oldId frame
        frame = self.landmarkManager.frames[oldId]
        for LandmarkId,ld in frame.observedLandmarks.items():
            observation = ld.observations[oldId]
            pointXYZ = ld.pointXYZ
            old3dPointsWorld.append(pointXYZ)
            old3dPointsLocal.append(pointWorldToCam(self.poses[oldId],pointXYZ))
            oldPoints.append(observation.pointUV)
        
        # get keypoints from oldId frame
        frame = self.landmarkManager.frames[curId]
        for LandmarkId,ld in frame.observedLandmarks.items():
            observation = ld.observations[curId]
            pointXYZ = ld.pointXYZ
            cur3dPointsWorld.append(observation.pointUV)
            cur3dPointsLocal.append(pointWorldToCam(self.poses[curId],pointXYZ))
            curPoints.append(observation.pointUV)




        ## Test 1: Test Pnp
        rvec = np.zeros(3,dtype=np.float)
        tvec = np.zeros((3*1),dtype=np.float)
        _, rvec, tvec = cv2.solvePnP(np.array(old3dPointsWorld),
                                     np.array(oldPoints).reshape(-1,2),
                                     self.K.reshape((3,3)),
                                     np.zeros(5))
            
        print(rvec)
        rot = cv2.Rodrigues(rvec)[0]
        
        
        pose = self.poses[oldId]
        T_GT = np.zeros((4,4))
        T_GT[0:3,0:3] = np.array(pose[0]).reshape(3,3)
        T_GT[0:3,3] = pose[1].T
        T_GT[3,3] = 1

        T_Cal = np.zeros((4,4))
        T_Cal[0:3,0:3] = rot
        T_Cal[0:3,3] = tvec.reshape(3)
        T_Cal[3,3] = 1
        T_Cal = np.linalg.inv(T_Cal)
        print("pnp result")
        print(T_Cal)
        print("ground truth")
        print(T_GT)



        







            
            


            



        


        

if __name__ == '__main__':
    FP = flowParser(Path)
    FP.mainThread()