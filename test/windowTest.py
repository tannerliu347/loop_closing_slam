import numpy as np
import glob 
import cv2
import matplotlib.pyplot as plt
from landmarkManager import LandmarkManager,Landmark,Observation,Frame
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import teaserpp_python
import time
Path = "/home/bigby/tartanair_tools/P001/"
NOISE_BOUND = 0.05
N_OUTLIERS = 1700
OUTLIER_TRANSLATION_LB = 5
OUTLIER_TRANSLATION_UB = 10
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
        T1 = np.eye(4)
        T1[0:3,0:3] = rot1
        T1[0:3,3] = t1.T
        T1[3,3] = 1
   
        T2 = np.eye(4)
        T2[0:3,0:3] = rot2
        T2[0:3,3] = t2.T
        T2[3,3] = 1
     
        Point = cv2.triangulatePoints(K@T1, K@T2, np.array(observation1.pointUV), np.array(observation2.pointUV))
        Point = Point/Point[3]
      
        return Point[0:3]
    def calculate_error(pred, gt,filename=''):
        totalerror = 0
        errors =[]
        for i in range(0,1):
            current_pred = pred
            current_gt = gt
            error = np.linalg.norm(np.linalg.inv(current_pred)@current_gt - np.identity(4), 'fro')
            errors.append(error)
            totalerror += error
        print(filename)
        print("error: " + str(totalerror))
        # np.savetxt(filename,np.array(errors).reshape(4,1), fmt='%.8f')
        return totalerror
    def pose_to_T(self,pose):
        t = pose[1]
        rot = np.array(pose[0]).reshape(3,3)
        T = np.eye(4)
        T[0:3,0:3] = rot
        T[0:3,3] = t.T
        T[3,3] = 1
        return T
    def mainThread(self,use_feature = True):
        curId = [190,210]
        # oldId = [238,239]
        #curId = [5,6]
        # oldId = [238,239]
        oldId = [230,250]
        # calculate feature 
        # if (use_feature == True):
        #     des_cur, point3d_cur = self.calculate_feature(curId)
        #     des_next, point3d_next = self.calculate_feature(oldId)
        # else:
        #     des_cur, point3d_cur,color_3d_cur = self.calculatePointCloudDepth(curId)
        #     des_next, point3d_next,color_3d_next = self.calculatePointCloudDepth(oldId)
        # pass
        # # self.save_as_pcd(point3d_cur)
        # # self.save_as_pcd(point3d_next)
        # print(point3d_cur)

        # get pose 
        poses_cur = {}
        for frame in curId:
            if frame == curId[0]:
                poses_cur[frame] = np.eye(4)
                continue
            p = self.poses[frame]
            T = self.pose_to_T(p)

            p0 = self.poses[curId[0]]
            T0 = self.pose_to_T(p0)
            poses_cur[frame] = np.linalg.inv(T0)@T
            print(np.linalg.inv(T0)@T)
        poses_old = {}
        for frame in oldId:
            if frame == oldId[0]:
                poses_old[frame] = np.eye(4)
                continue
            p = self.poses[frame]
            T = self.pose_to_T(p)
            p0 = self.poses[oldId[0]]
            T0 = self.pose_to_T(p0)
            # poses_cur[frame] = np.linalg.inv(T0)@T
            poses_old[frame] = np.linalg.inv(T0)@T
        pcd1 = self.pcd_transform(poses_cur,curId)
        pcd2 = self.pcd_transform(poses_old,oldId)



        # do registration use icp 
        # pcd1 = o3d.geometry.PointCloud()
        # pcd1.points = o3d.utility.Vector3dVector(point3d_cur)
        # pcd2 = o3d.geometry.PointCloud()
        # pcd2.points = o3d.utility.Vector3dVector(point3d_next)
        pcd1.paint_uniform_color([1, 0.706, 0])
        # pcd2.paint_uniform_color([0, 0.651, 0.929])
        # print(color_3d_cur.shape)
        # pcd1.colors = o3d.utility.Vector3dVector(color_3d_cur)
        # pcd2.colors = o3d.utility.Vector3dVector(color_3d_next)
        # o3d.visualization.draw_geometries([pcd2])
        print(pcd1)
        print(pcd2)
        # evaluate normal of pcd1 and pcd2
        # pcd1.estimate_normals(
        # search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.5, max_nn=30))
        # pcd2.estimate_normals(
        # search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.5, max_nn=30))
        
        # print("Initial alignment")
        # evaluation = o3d.pipelines.registration.evaluate_registration(pcd1, pcd2, 0.01)
        # print(evaluation)
        print("Apply point-to-point ICP")
        reg_p2p = o3d.pipelines.registration.registration_icp( pcd1, pcd2, 0.01, np.eye(4),o3d.pipelines.registration.TransformationEstimationPointToPoint())
        print(reg_p2p)
        print("Transformation is:")
        print(reg_p2p.transformation)
        pcd2.transform(reg_p2p.transformation)
        # o3d.visualization.draw_geometries([pcd1, pcd2])


        # Populating the parameters
        solver_params = teaserpp_python.RobustRegistrationSolver.Params()
        solver_params.cbar2 = 1
        solver_params.noise_bound = NOISE_BOUND
        solver_params.estimate_scaling = False
        solver_params.rotation_estimation_algorithm = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
        solver_params.rotation_gnc_factor = 1.4
        solver_params.rotation_max_iterations = 100
        solver_params.rotation_cost_threshold = 1e-12
        point_current = np.asarray(pcd1.points).T
        point_next = np.asarray(pcd2.points).T
        print((point_current).shape)
        point_next = point_next[:,0:1070]
        # print(point_next)
        print((point_next).shape)
        solver = teaserpp_python.RobustRegistrationSolver(solver_params)
        start = time.time()
        solver.solve(point_current, point_next)
        end = time.time()
        solution = solver.getSolution()
        print(point_current.shape)
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(point_current.T)
        T = np.eye(4)
        T[0:3,0:3] = solution.rotation
        T[0:3,3] = solution.translation
        pcd1.transform(T)
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(point_next.T)
        pcd1.paint_uniform_color([1, 0.706, 0])
        pcd2.paint_uniform_color([0, 0.651, 0.929])
        o3d.visualization.draw_geometries([pcd1, pcd2])
        print("")
    def pcd_transform(self,poses,window):
        pcd_combined = o3d.geometry.PointCloud()
        for frame in window:
            # generate random color use numpy
            color = np.random.rand(3)
            des_cur, point3d_cur,color_3d_cur = self.calculatePointCloudDepth([frame])
            pcd =   o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point3d_cur)
            pcd.paint_uniform_color(color)
            r = R.from_euler('x', -90, degrees=True).as_matrix() @   R.from_euler('z', -90, degrees=True).as_matrix()
            T_r = np.eye(4)
            T_r[0:3,0:3] = r
            pcd.transform(poses[frame] @ np.linalg.inv(T_r))
            pcd_combined += pcd
        return pcd_combined

    
    def save_as_pcd(self,points):
        # display python array as pcd file
        pcd_points = o3d.geometry.PointCloud()
        pcd_points.points = o3d.utility.Vector3dVector(points)
        o3d.visualization.draw_geometries([pcd_points])
    def calculatePointCloudDepth(self,window):
        depth = self.depthMap[window[0]]
        image = self.imageMap[window[0]]
        # detect feature use sift 
        surf = cv2.xfeatures2d.SIFT_create(30000)
        kp, des = surf.detectAndCompute(image,None)
        # display keypoint use opencv
        img = cv2.drawKeypoints(image,kp,None,(255,0,0),4)
        cv2.imshow('keypoint',img)
        # cv2.waitKey(0)
        px = self.K[0,2]
        fx = self.K[0,0]
        py = self.K[1,2]
        fy = self.K[1,1]
        point_3d = []
        color_3d = []
        for pt in kp:
            u = int(pt.pt[0])
            v = int(pt.pt[1])
            # print(u,v)
            # if (u < 0 or v < 0):
            #         continue
            if (u > self.imageH  or v > self.imageW):
                    continue
            # convert pixel coordinate to camera coordinate use depth at that pixel
            d = depth[int(v)][int(u)]
            if (d != 0 and d < 100):
                x = (u-px)*d/fx
                y = (v-py)*d/fy
                point_3d.append([x,y,d])
                color_3d.append(image[int(v)][int(u)])
        return des,point_3d,np.array(color_3d)
                
    def calculate_feature(self,window):
        self.matches_id_map = {} 
        self.matches = []
        self.nextKps = []
        # detect landmark 
        if window[0] not in self.imageMap or  window[1] not in self.imageMap:
            return False
        image = self.imageMap[window[0]]
        image_next = self.imageMap[window[1]]
        flow = self.flowMap[window[0]]
        flow_next =   self.flowMap[window[1]]
        flowMask = self.flowmaskMap[window[0]]
        flowMask_next = self.flowmaskMap[window[1]]
        depth = self.depthMap[window[0]]
        depth_next = self.depthMap[window[1]]
        # Get feature from image use opencv 
        sift = cv2.xfeatures2d.SIFT_create(30000)
        currentKps, des1 = sift.detectAndCompute(image,None)
        

        kp_next = []
        kp_cur = []
        index_cur = 0
        index_next = 0
        for kp in currentKps:
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
        
        # Remove untrackablkey point 
        
        matched_image = cv2.drawMatches(image, kp_cur, image_next, kp_next, self.matches , None, flags=2)
        cv2.imshow('SIFT', matched_image)
        # cv2.waitKey(0)

        # do triangulation and add 3d point in landmark manager
        point_3d = []
        for i in range(len(kp_cur)):
            depth_cur = depth[int(kp_cur[i].pt[1]),int(kp_cur[i].pt[0])]
            depth_Ne = depth_next[int(kp_next[i].pt[1]),int(kp_next[i].pt[0])]
            newObservationCur = Observation(window[0],kp_cur[i].pt,depth_cur)
            newObservationNe = Observation(window[1],kp_next[i].pt,depth_Ne)
            

            pointxyz = self.triangulation(self.poses[window[0]],self.poses[window[1]],newObservationCur,newObservationNe)
            point_3d.append(pointxyz)
        return des1,point_3d
if __name__ == '__main__':
    FP = flowParser(Path)
    FP.mainThread(False)
    