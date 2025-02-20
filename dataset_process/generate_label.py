import numpy as np
import trimesh
import torch
from pysdf import SDF  # 使用 pysdf 计算 SDF
import mesh_to_sdf
from typing import Tuple
from skimage import measure
import open3d as o3d
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

# import pyrender
#
# # 禁用窗口创建
# renderer = pyrender.OffscreenRenderer(1, 1)

def normalize_mesh(mesh: trimesh.Trimesh) -> Tuple[float, np.ndarray]:
    """Normalizes the mesh by scaling and translating it to fit in a unit cube centered at the origin."""
    # 获取原始包围盒
    bbox_min, bbox_max = mesh.bounds
    scale = 1 / np.max(bbox_max - bbox_min)  # 计算缩放因子，使最长边为1
    mesh.apply_scale(scale)

    # 更新缩放后的包围盒
    bbox_min, bbox_max = mesh.bounds
    center = (bbox_min + bbox_max) / 2
    offset = -center  # 计算平移量：将中心移到原点，因此取负值
    mesh.apply_translation(offset)

    return scale, offset


def generate_sampling_points(density: int) -> torch.Tensor:
    """Generates 3D sampling points within the range [-1, 1]^3."""
    x_coor = np.linspace(-0.5, 0.5, density + 1)
    y_coor = np.linspace(-0.5, 0.5, density + 1)
    z_coor = np.linspace(-0.5, 0.5, density + 1)

    # Generate a grid of points
    xv, yv, zv = np.meshgrid(x_coor, y_coor, z_coor, indexing='ij')
    queries = torch.from_numpy(np.stack([xv, yv, zv], axis=0).astype(np.float32))
    queries = queries.view(3, -1).transpose(0, 1)[None]  # Shape: [1, N, 3]

    return queries


def check_points_sdf(mesh: trimesh.Trimesh, points: torch.Tensor) -> np.ndarray:
    """Calculates the Signed Distance Function (SDF) for each point."""
    points_np = points.squeeze(0).cpu().numpy()

    # # 将 mesh 转换为 pysdf 的格式 (顶点和面)
    # f = SDF(mesh.vertices, mesh.faces)
    # # 将 torch tensor 转换为 numpy 数组并计算 SDF
    # sdf_values = f(points_np)  # 计算每个点的 SDF 值


    print("extracting sdf")
    sdf_values = mesh_to_sdf.mesh_to_sdf(mesh, points_np)
    print("success")

    return sdf_values


def save_labels(labels: np.ndarray, density: int, output_path: str):
    """Saves the labels to a .npy file."""
    labels_3d = labels.reshape((density + 1, density + 1, density + 1))
    np.save(output_path, labels_3d)
    print(f"Saved labels to {output_path} with shape {labels_3d.shape}")


def process_mesh_and_points(mesh_path: str, density: int, output_path: str):
    """Full processing pipeline: Normalize mesh, generate points, calculate SDF, and save results."""
    # Load mesh using trimesh
    mesh = trimesh.load(mesh_path)

    # Normalize the mesh (scale and translate to fit inside a unit cube centered at the origin)
    normalize_mesh(mesh)

    mesh = mesh.to_mesh()
    # o3d.io.write_triangle_mesh("ori_mesh.obj", mesh)
    # mesh.export("ori_mesh.obj")

    print(f"mesh is watertight: {mesh.is_watertight}")
    # print(len(trimesh.repair.broken_faces(mesh, color=[255,0,0,255])))

    mesh.export("mc_mesh_ori.obj")

    # Generate sampling points
    points = generate_sampling_points(density)

    # Check if points are inside the mesh (by computing their SDF)
    labels = check_points_sdf(mesh, points)

    # Save the labels
    save_labels(labels, density, output_path)

    sdf = labels.reshape((129, 129, 129))

    # 计算网格间隔，129 个点覆盖 [-0.5, 0.5]，故间隔为 1/128
    spacing = (1.0 / 128, 1.0 / 128, 1.0 / 128)

    # 使用 marching_cubes，并指定 spacing 参数
    verts, faces, normals, values = measure.marching_cubes(sdf, level=0, spacing=spacing)

    # 由于 marching_cubes 返回的顶点坐标基于 [0, 1]，需要平移到 [-0.5, 0.5]
    verts += np.array([-0.5, -0.5, -0.5])

    # 构造 Open3D 三角网格并显示
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    o3d.io.write_triangle_mesh("mc_mesh.obj", mesh)


# Example usage:
mesh_path = '/mnt/merged_nvme/lht/TRELLIS/objaverse_sketchfab/raw/hf-objaverse-v1/glbs/000-072/8e95e24a076d4cb39519defe71d3b18b.glb'  # Replace with the path to your mesh file
# mesh_path = "cube.obj"
print(f"reading: {mesh_path}")
density = 128  # Define your density
output_path = 'output_labels.npy'  # Output path for the saved labels

process_mesh_and_points(mesh_path, density, output_path)



print()
