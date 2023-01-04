import numpy as np
import matplotlib.pyplot as plt
import gtsam
from gtsam.utils import plot
# print(matplotlib.__version__)
from gtsam import Point3, Pose3, Rot3
from gtsam.symbol_shorthand import X
# gtsam.utils.plot.plot_trajectory(0,initial)

class LandmarkManager:
    def __init__(self):
        self.landmarks = {}
        self.frames = {}
        self.cameraMatrix = None
        self.cameraPose = None
class Observation:
    def __init__(self,frameId,pointUV,depth = -1):
        self.framId = frameId
        self.pointUV = pointUV
        self.depth = depth
    # def __init__(self,frameId,pointUV):
    #     self.framId = frameId
    #     self.pointUV = pointUV
       
class Landmark:
    def __init__(self,landmarkId):
        self.landmarkId = landmarkId
        self.pointXYZ = None
        self.observedFrames = {}
        self.observations = {}
class Frame:
    def __init__(self,frameId):
        self.frameId = frameId
        self.connectedIds = []
        self.observedLandmarks = {}