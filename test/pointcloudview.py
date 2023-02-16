import numpy as np
import copy
import open3d as o3
from probreg import cpd
import time
# load source and target point cloud

# target = copy.deepcopy(source)
# transform target point cloud
# th = np.deg2rad(30.0)
# target.transform(np.array([[np.cos(th), -np.sin(th), 0.0, 0.0],
#                            [np.sin(th), np.cos(th), 0.0, 0.0],
#                            [0.0, 0.0, 1.0, 0.0],
#                            [0.0, 0.0, 0.0, 1.0]]))


vis = o3.visualization.Visualizer()
vis.create_window()
target = o3.io.read_point_cloud('/home/bigby/curly_slam/catkin_ws/pc/1.pcd')
target.paint_uniform_color([1, 0, 0])
vis.add_geometry(target)
for i in range(2,105):
    print(i)
    filename = '/home/bigby/curly_slam/catkin_ws/pc' + str(i) + '.pcd'
    target = o3.io.read_point_cloud(filename) 
    # vis.clear_geometries()
    vis.add_geometry(target)
    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.2)