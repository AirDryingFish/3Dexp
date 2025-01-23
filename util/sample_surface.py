import os
import pandas as pd
import trimesh
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def sample_points_and_normals_from_mesh(glb_file_path, num_points=10000):
    """
    从一个 .glb 文件中随机采样表面点云及对应的法线。

    Args:
        glb_file_path (str): .glb 文件路径。
        num_points (int): 采样点的数量。

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - points: (num_points, 3) 的点云数组。
            - normals: (num_points, 3) 的法线数组。
    """
    # 加载 .glb 文件
    mesh = trimesh.load(glb_file_path)

    if isinstance(mesh, trimesh.Scene):
        # 如果加载的是一个场景，合并所有子网格
        mesh = trimesh.util.concatenate(mesh.dump())

    # 确保是三角网格
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("输入的文件未解析为有效的三角网格")

    # 从网格中随机采样点和法线
    sampled_points, face_indices = trimesh.sample.sample_surface(mesh, num_points)
    normals = mesh.face_normals[face_indices]

    return sampled_points, normals

def process_file(row, data_root, points_root, metadata, index):
    """
    处理单个文件，采样点云并保存为 .npz。
    """
    local_path = row['local_path']  # 获取文件路径
    full_path = os.path.join(data_root, local_path)  # 拼接完整路径

    if os.path.exists(full_path):
        try:
            # 使用 trimesh 采样点和法线
            sampled_points, normals = sample_points_and_normals_from_mesh(full_path)

            sha256_name = local_path.split('/')[-1].replace(".glb", ".npz")
            group_num = local_path.split('/')[-2]
            # 构造 .npz 保存路径，保持与原 local_path 结构一致
            npz_path = os.path.join(points_root, group_num, sha256_name)
            os.makedirs(os.path.dirname(npz_path), exist_ok=True)  # 创建文件夹

            # 保存为 .npz 文件
            np.savez(npz_path, points=sampled_points, normals=normals)

            # 在 metadata 中记录路径
            metadata.at[index, 'pointcloud_path'] = npz_path
        except Exception as e:
            print(f"文件加载失败: {full_path}, 错误: {e}")
    else:
        print(f"文件不存在: {full_path}")

# 加载 metadata.csv 文件
data_root = "/mnt/merged_nvme/lht/TRELLIS/objaverse_sketchfab/"  # 替换为数据集根目录的绝对路径
csv_path = os.path.join(data_root, "metadata.csv")

# 创建存储点云数据的根目录
points_root = os.path.join(data_root, "surface_points")

# 读取 CSV
metadata = pd.read_csv(csv_path)

# 新增一列记录 .npz 文件路径
metadata['pointcloud_path'] = ""

# 使用 ThreadPoolExecutor 进行多线程处理
with ThreadPoolExecutor(max_workers=8) as executor:
    # 创建进度条
    with tqdm(total=len(metadata), desc="Processing Files") as pbar:
        # 提交任务到线程池
        futures = {
            executor.submit(process_file, row, data_root, points_root, metadata, index): index
            for index, row in metadata.iterrows()
        }

        # 遍历已完成的任务并更新进度条
        for future in futures:
            future.result()  # 等待任务完成
            pbar.update(1)  # 每完成一个任务，更新进度条

# 保存更新后的 metadata.csv 文件
updated_csv_path = os.path.join(data_root, "updated_metadata.csv")
metadata.to_csv(updated_csv_path, index=False)
print(f"更新后的 CSV 文件已保存到: {updated_csv_path}")
