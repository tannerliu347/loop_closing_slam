import numpy as np
import glob 
import cv2
import matplotlib.pyplot as plt
from landmarkManager import LandmarkManager,Landmark,Observation,Frame
Path = "/home/bigby/Downloads/tartanair_tools-master/seasidetown/Easy/seasidetown/seasidetown/Easy/P002/"


class flowParser():
    def __init__(self,flowPath):
        # Load flow mask
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
        
        self.currentKps = []
        self.nextKps = []
        self.matches = []
        self.curGlobalIDs= []
        self.matches_id_map = {}
        print(self.imageMap[0].shape)
        self.frameID = 0
        self.landmarkId = 0
        self.image  = self.imageMap[0]
        self.flowMask = self.flowmaskMap[0]
        self.flow = self.flowMap[0]
        self.frameGap = 1
        self.landmarkManager = LandmarkManager()

        # Todo: multi frame flow track
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
                    newObservationCur = Observation(self.frameID,self.currentKps[cur_index].pt)
                    newObservationNe = Observation(self.frameID + self.frameGap,self.nextKps[i].pt)
                    newLandmark.observations[self.frameID] = newObservationCur
                    newLandmark.observations[self.frameID + self.frameGap] = newObservationNe
                    self.landmarkManager.landmarks[globalId]= newLandmark
                    self.landmarkManager.frames[self.frameID + 1].observedLandmarks[globalId] = self.landmarkManager.landmarks[globalId]
                    self.landmarkManager.frames[self.frameID].observedLandmarks[globalId] = self.landmarkManager.landmarks[globalId]
            else:
                globalId = self.curGlobalIDs[cur_index]
                # old Point only save next 
                newObservationNe = Observation(self.frameID + self.frameGap,self.nextKps[i].pt)
                self.landmarkManager.landmarks[globalId].observations[self.frameID + self.frameGap] = newObservationNe
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
            # self.createKeyframe()
            
            # Update image, flow, and flow mask
            try:
                self.image  = self.imageMap[self.frameID]
                self.flowMask = self.flowmaskMap[self.frameID]
                self.flow = self.flowMap[self.frameID]
            except:
                print("all frame parsed")
                self.saveFrontendResult(self.frameID)
                return
            

        # update pointID and calculate triangulation Point 

        # 
    def saveFrontendResult(self,lastFrameId):
        f = open("groundtruthflowFrontend.txt",'w')
        f.write("FORMAT 1\n")
        f.write("CAMERA 320 320 320 240\n")
        for frameID in range(lastFrameId):
            f.write("FRAME " + str(frameID) +"\n")
            frame = self.landmarkManager.frames[frameID]
            for LandmarkId,ld in frame.observedLandmarks.items():
                f.write("FEATURE " + str(LandmarkId) + "\n")


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
                

            



        


        

if __name__ == '__main__':
    FP = flowParser(Path)
    FP.mainThread()