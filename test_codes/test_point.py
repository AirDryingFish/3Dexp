import numpy as np
import os
import trimesh
import matplotlib.pyplot as plt
data_dir = "/home/yangzengzhi/data_lht/3D/ShapeNetV2_watertight/02691156/4_pointcloud"
file_dir = os.path.join(data_dir, "879ebdd198cf4aa58f6810e1a2b6aa04.npz")
data = np.load(file_dir)
points, normals = data['points'], data['normals']
cloud = trimesh.points.PointCloud(points, normals=normals)
# vol_points = data['vol_points']
mesh = cloud.convex_hull

mesh.show()
# mesh.export('/home/yangzengzhi/test_output/test2.ply')
# print(data['points'])