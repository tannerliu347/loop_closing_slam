import numpy as np
import copy
import open3d as o3
from probreg import cpd

# load source and target point cloud
source = o3.io.read_point_cloud('/home/bigby/curly_slam/catkin_ws/pc/15.pcd')
source.remove_non_finite_points()
print(source)
# target = copy.deepcopy(source)
# transform target point cloud
# th = np.deg2rad(30.0)
# target.transform(np.array([[np.cos(th), -np.sin(th), 0.0, 0.0],
#                            [np.sin(th), np.cos(th), 0.0, 0.0],
#                            [0.0, 0.0, 1.0, 0.0],
#                            [0.0, 0.0, 0.0, 1.0]]))
target = o3.io.read_point_cloud('/home/bigby/curly_slam/catkin_ws/pc/104.pcd')
print(target)

source.remove_non_finite_points()
# source = source.voxel_down_sample(voxel_size=0.005)
# target = target.voxel_down_sample(voxel_size=0.005)
source.paint_uniform_color([1, 0, 0])
target.paint_uniform_color([0, 1, 0])
o3.visualization.draw_geometries([target, source])
# compute cpd registration
tf_param, _, _ = cpd.registration_cpd(source, target)
result = copy.deepcopy(source)
result.points = tf_param.transform(result.points)
# draw result
source.paint_uniform_color([1, 0, 0])
target.paint_uniform_color([0, 1, 0])
o3.visualization.draw_geometries([target, result])