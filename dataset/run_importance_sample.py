import concurrent.futures
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import trimesh
from trimesh.util import concatenate
import open3d as o3d
from trimesh.boolean import union
# def load_and_triangulate(mesh_path):
#     loaded = trimesh.load(mesh_path, force='mesh', skip_missing=True)
#     if isinstance(loaded, trimesh.Scene):
#         sub_meshes = list(loaded.geometry.values())
#         big_mesh = concatenate(sub_meshes)
#     else:
#         big_mesh = loaded
#
#     # big_mesh = big_mesh.triangulate()  # 如需三角化可打开
#     big_mesh.merge_vertices()
#     big_mesh.remove_unreferenced_vertices()
#     return big_mesh

def merge_outer_shell_by_union(mesh_or_scene):
    """
    若加载到的是多个子mesh (场景), 利用 boolean.union 合并成单一外壳。
    """
    # 如果已经是Trimesh就直接返回
    if isinstance(mesh_or_scene, trimesh.Trimesh):
        return mesh_or_scene

    # 否则说明是Trimesh.Scene
    sub_meshes = []
    for geom in mesh_or_scene.geometry.values():
        # 只收集真正的Trimesh对象
        if isinstance(geom, trimesh.Trimesh):
            sub_meshes.append(geom)

    if not sub_meshes:
        return None

    # 对这些子mesh做布尔并 => 得到一个Trimesh
    # 注意: 大网格 + engine='scad' 可能很慢
    result = union(sub_meshes, engine='scad')
    return result

def load_outer_shell_via_union(mesh_path):
    """
    载入多子mesh场景 => 做布尔并 => 返回单一外壳
    """
    loaded = trimesh.load(mesh_path, force='mesh', skip_missing=True)
    if isinstance(loaded, trimesh.Scene):
        merged = merge_outer_shell_by_union(loaded)
        # 对合并结果做清理
        if merged is not None:
            merged.merge_vertices()
            merged.remove_unreferenced_vertices()
        return merged
    else:
        # 单一Trimesh就不用布尔并
        loaded.merge_vertices()
        loaded.remove_unreferenced_vertices()
        return loaded


def angle_between_normals(nA, nB, eps=1e-14):
    """
    计算两个法线向量nA,nB的夹角[0,pi]，并做数值保护。
    """
    normA = np.linalg.norm(nA)
    normB = np.linalg.norm(nB)
    if normA < eps or normB < eps:
        return 0.0
    dot_val = np.dot(nA, nB) / (normA * normB)
    dot_val = max(min(dot_val,1.0), -1.0)
    return np.arccos(dot_val)

def find_sharp_edges_and_edge_map(mesh, angle_threshold_deg=30.0):
    """
    返回:
      sharp_edges: [(v0,v1), ...] (满足二面角阈值的锐边)
      edge_face_map: { (v0,v1): [face_idx,...], ... } (记录每条边相邻的面)
    """
    faces = mesh.faces
    face_normals = mesh.face_normals
    edge_face_map = {}

    for f_idx, f in enumerate(faces):
        vs = [f[0], f[1], f[2]]
        es = [tuple(sorted((vs[0], vs[1]))),
              tuple(sorted((vs[1], vs[2]))),
              tuple(sorted((vs[0], vs[2])))]
        for e in es:
            if e not in edge_face_map:
                edge_face_map[e] = []
            edge_face_map[e].append(f_idx)

    angle_threshold_rad = np.deg2rad(angle_threshold_deg)
    sharp_edges = []

    for edge, f_list in edge_face_map.items():
        # 如果此边只属于1个面(边界)或其他情况
        if len(f_list) < 2:
            continue

        # 找出此边相邻面的最大夹角
        max_angle = 0.0
        for i in range(len(f_list)):
            for j in range(i+1, len(f_list)):
                fA = f_list[i]
                fB = f_list[j]
                nA = face_normals[fA]
                nB = face_normals[fB]
                angle = angle_between_normals(nA, nB)
                if angle>max_angle:
                    max_angle = angle

        if max_angle>angle_threshold_rad:
            sharp_edges.append(edge)

    return sharp_edges, edge_face_map

def sample_points_on_sharp_edges(mesh, sharp_edges, samples_per_edge=5):
    """
    在每条锐边上做线性等分采样。
    """
    vertices = mesh.vertices
    result = []
    for edge in sharp_edges:
        v0, v1 = edge
        V0 = vertices[v0]
        V1 = vertices[v1]
        for i in range(samples_per_edge+1):
            t = i/samples_per_edge
            p = V0 + t*(V1 - V0)
            result.append(p)
    if len(result)==0:
        return np.zeros((0,3))
    return np.vstack(result)

def point_to_segment_distance(p, v0, v1):
    """
    计算点p到线段[v0,v1]的最小距离。
    """
    seg = v1 - v0
    denom = np.dot(seg, seg)
    if denom<1e-14:
        return np.linalg.norm(p - v0)
    t = np.dot(p - v0, seg)/denom
    if t<0:
        closest = v0
    elif t>1:
        closest = v1
    else:
        closest = v0 + t*seg
    return np.linalg.norm(p-closest)

def sample_points_on_adjacent_faces_biased_to_sharp_edges(
        mesh,
        sharp_edges,
        edge_face_map,
        num_samples=20000,
        beta=0.5,
        max_iter_factor=5,
        dist_eps=1e-4,
        r_threshold=5.0
):
    """
    只在包含锐边的面上进行采样。
    当点到锐边距离>r_threshold时，不算浪费一次计数，继续随机生成点。
    并对距离<=r_threshold的候选点，用概率(越近越大)判断是否保留。
    【新增】如果在指定次数内无法找到点，就“无条件”补一个随机三角形采样点，保证面采样数达标。
    """
    from numpy import random

    faces = mesh.faces
    vertices = mesh.vertices
    face_areas = mesh.area_faces

    # 1) 找出所有包含锐边的三角面
    adjacent_faces = set()
    for e in sharp_edges:
        flist = edge_face_map[e]
        for f_idx in flist:
            adjacent_faces.add(f_idx)

    if not adjacent_faces:
        return np.zeros((0, 3))

    # 2) 只给这些面分配采样
    area_adj = sum(face_areas[f] for f in adjacent_faces)
    if area_adj < 1e-14:
        return np.zeros((0, 3))

    face_samples = {}
    for f_idx in adjacent_faces:
        frac = face_areas[f_idx] / area_adj
        face_samples[f_idx] = int(round(num_samples * frac))

    # 3) face->锐边映射
    sharp_edges_set = set(sharp_edges)
    face_edge_map_local = [[] for _ in range(len(faces))]
    edge_coord_map = {}
    for e in sharp_edges:
        v0, v1 = e
        edge_coord_map[e] = (vertices[v0], vertices[v1])

    for i, f in enumerate(faces):
        vs = [f[0], f[1], f[2]]
        es = [
            tuple(sorted((vs[0], vs[1]))),
            tuple(sorted((vs[1], vs[2]))),
            tuple(sorted((vs[0], vs[2])))
        ]
        edges_sharp = [e for e in es if e in sharp_edges_set]
        face_edge_map_local[i] = edges_sharp

    all_points = []

    # 4) 对每个面做"双重循环"采样
    from numpy import random
    for f_idx in adjacent_faces:
        needed = face_samples[f_idx]
        if needed <= 0:
            continue

        tri = faces[f_idx]
        v1, v2, v3 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]

        edges_sharp = face_edge_map_local[f_idx]

        # 若该面没有锐边(但被标记到adjacent_faces，可能多面共享)，就普通采样
        if len(edges_sharp) == 0:
            r1 = random.rand(needed)
            r2 = random.rand(needed)
            mask = (r1 + r2) > 1.0
            r1[mask] = 1.0 - r1[mask]
            r2[mask] = 1.0 - r2[mask]
            p = v1 + (v2 - v1) * r1[:, None] + (v3 - v1) * r2[:, None]
            all_points.append(p)
            continue

        # 含锐边 => 距离+概率的贴边逻辑
        sharp_edge_pairs = [edge_coord_map[e] for e in edges_sharp]

        collected = []
        max_total_attempts = needed * max_iter_factor
        attempts_used = 0

        for _ in range(needed):
            got_one = False
            # 在这里不停生成候选点,尝试贴边距离+概率通过
            while attempts_used < max_total_attempts:
                attempts_used += 1

                # 生成barycentric随机
                r1 = random.rand()
                r2 = random.rand()
                if (r1 + r2) > 1.0:
                    r1, r2 = 1.0 - r1, 1.0 - r2
                cand = v1 + (v2 - v1) * r1 + (v3 - v1) * r2

                # 计算cand到锐边的最小距离
                min_d = float('inf')
                for (ev0, ev1) in sharp_edge_pairs:
                    d = point_to_segment_distance(cand, ev0, ev1)
                    if d < min_d:
                        min_d = d

                # 若 > r_threshold => 跳过（不消耗外层count）
                if min_d > r_threshold:
                    continue

                # 否则 => 根据距离贴边概率
                w_cand = 1.0 / (min_d + dist_eps)
                p_accept = 1.0 - np.exp(-beta * w_cand)
                if random.rand() < p_accept:
                    # 成功
                    collected.append(cand)
                    got_one = True
                    break

            # 如果在 max_total_attempts 内也没找到
            if not got_one:
                # 新增：执行“无条件”补一个随机点(不考虑r_threshold)
                r1 = random.rand()
                r2 = random.rand()
                if r1 + r2 > 1.0:
                    r1, r2 = 1.0 - r1, 1.0 - r2
                cand = v1 + (v2 - v1) * r1 + (v3 - v1) * r2
                collected.append(cand)

        if len(collected) > 0:
            all_points.append(np.array(collected))

    if not all_points:
        return np.zeros((0, 3))

    return np.vstack(all_points)

def farthest_point_downsample(edge_pts, num_samples):
    """
    使用Open3D的farthest_point_down_sample对给定的edge_pts进行下采样
    :param edge_pts: 输入的点云数据，类型为ndarray
    :param num_samples: 要下采样的点的数量
    :return: 下采样后的点云数据，类型为ndarray
    """
    # 将ndarray转换为open3d.geometry.PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(edge_pts)

    # 使用farthest_point_down_sample进行下采样
    downsampled_pcd = pcd.farthest_point_down_sample(num_samples)

    # 获取下采样后的点
    downsampled_points = np.asarray(downsampled_pcd.points)

    return downsampled_points


def uniform_downsample_points(points_ndarray, num_samples):
    """
    使用Open3D的uniform_down_sample函数对给定的numpy点云进行均匀下采样。

    参数:
      points_ndarray (np.ndarray): 原始点云，形状(N, 3)
      every_k_points (int): 下采样步长。选取 [0, k, 2k, ...] 这些下标的点。

    返回:
      np.ndarray: 下采样后的点云，形状(M, 3)
    """
    every_k_points = points_ndarray.shape[0] // num_samples

    # 1) 将NumPy数组转换为Open3D的PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_ndarray)

    # 2) 调用uniform_down_sample进行均匀下采样
    pcd_down = pcd.uniform_down_sample(every_k_points)

    # 3) 将下采样结果转换回NumPy数组
    downsampled_points = np.asarray(pcd_down.points)

    return downsampled_points

def run_sharp_edge_sampling(
    mesh_path,
    angle_threshold_deg=30.0,
    edge_samples_per_edge=10,
    face_samples=20000,
    beta=0.5,
    r_threshold_ratio=0.05,  # 用包围盒对角线的比例
    max_iter_factor=5
):
    """
    主函数:
      - angle_threshold_deg: 二面角阈值
      - edge_samples_per_edge: 每条锐边上线性等分采样数
      - face_samples: 面上总采样数量(含锐边面)
      - beta: 贴边敏感度(越大越贴)
      - r_threshold_ratio: 距离阈值占对角线的比例(0~1或更大)
      - max_iter_factor: 内部拒绝采样时,每个点最多尝试多少倍(needed)
    """
    # print(f"Loading mesh from: {mesh_path}")
    mesh = load_outer_shell_via_union(mesh_path)
    # print(f"[DEBUG] {mesh_path} => faces={len(mesh.faces)}, verts={len(mesh.vertices)}")
    # 根据包围盒对角线归一化r_threshold
    min_corner, max_corner = mesh.bounds
    diag_len = np.linalg.norm(max_corner - min_corner)
    r_threshold = diag_len * r_threshold_ratio
    # print(f"[INFO] diag_len={diag_len:.3f}, ratio={r_threshold_ratio}, => r_threshold={r_threshold:.3f}")

    # print("Finding sharp edges...")
    sharp_edges, edge_face_map = find_sharp_edges_and_edge_map(mesh, angle_threshold_deg)
    # print(f"  => Found {len(sharp_edges)} sharp edges above {angle_threshold_deg} deg")

    # print("Sampling from sharp edges...")
    edge_pts = sample_points_on_sharp_edges(mesh, sharp_edges, edge_samples_per_edge)
    # print(f"  => edge_pts: {edge_pts.shape[0]}")
    if face_samples > edge_pts.shape[0]:
        # print("Sampling from faces (double-loop, ensuring enough points if possible)...")
        face_pts = sample_points_on_adjacent_faces_biased_to_sharp_edges(
            mesh,
            sharp_edges,
            edge_face_map,
            num_samples=face_samples - edge_pts.shape[0],
            beta=beta,
            max_iter_factor=max_iter_factor,
            dist_eps=1e-4,
            r_threshold=r_threshold  # 使用归一化后的阈值
        )
        # print(f"  => face_pts: {face_pts.shape[0]}")
        all_points = np.vstack([edge_pts, face_pts])
        # print(f"Total points: {all_points.shape[0]}")
        return all_points
    else:
        # print("Running FPS sampling...")
        edge_pts = farthest_point_downsample(edge_pts, face_samples)
        all_points = edge_pts
        # print(f"Total points: {all_points.shape[0]}")
        return all_points


# ========================= 以下是并发处理逻辑，与 Poisson 示例类似 =========================
def process_one_model(row):
    """
    针对 metadata.csv 中一行(row)，执行 run_sharp_edge_sampling 后保存点云，返回信息：
      {
        "sha256": sha256_id,
        "importance_sample_path": 保存点云的路径,
        "importance_sample_count": 采样到的点数
      }
    """
    base_path = "/mnt/merged_nvme/lht/TRELLIS/objaverse_sketchfab/"
    mesh_file = os.path.join(base_path, row["local_path"])
    sha256_id = row["sha256"]

    # 自定义保存路径
    out_file = f"/mnt/merged_nvme/lht/TRELLIS/objaverse_sketchfab/importance_samples/{sha256_id}.ply"

    result_dict = {
        "sha256": sha256_id,
        "importance_sample_path": "",
        "importance_sample_count": 0
    }

    # 一些可调参数:
    angle_threshold_deg = 30.0
    edge_samples_per_edge = 2
    face_samples = 23000
    beta = 0.01
    r_threshold_ratio = 0.01  # 距离阈值=5%对角线
    max_iter_factor = 5

    try:
        # 执行锐边重要性采样
        points = run_sharp_edge_sampling(
            mesh_file,
            angle_threshold_deg=angle_threshold_deg,
            edge_samples_per_edge=edge_samples_per_edge,
            face_samples=face_samples,
            beta=beta,
            r_threshold_ratio=r_threshold_ratio,
            max_iter_factor=max_iter_factor
        )
        if points.shape[0] == 0:
            print(f"[DEBUG] Loading: {mesh_file}, exists={os.path.exists(mesh_file)}")
            print(f"Warning: no points sampled from {mesh_file}")
        else:
            # 保存采样结果
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            pcd = trimesh.points.PointCloud(points)
            pcd.export(out_file)

        # 更新结果
        result_dict["importance_sample_path"] = out_file
        result_dict["importance_sample_count"] = points.shape[0]

    except Exception as e:
        print(f"Error in importance sampling for {mesh_file}: {e}", flush=True)

    return result_dict


def main():
    # 原始的 metadata.csv 文件，包含 sha256 和 local_path 等
    csv_metadata_path = "/mnt/merged_nvme/lht/TRELLIS/objaverse_sketchfab/metadata.csv"
    df_meta = pd.read_csv(csv_metadata_path)

    # 确保至少包含 "sha256" 和 "local_path"
    if "sha256" not in df_meta.columns or "local_path" not in df_meta.columns:
        print("metadata.csv must contain 'sha256' and 'local_path' columns!")
        return

    # 读取或创建 importance_sampled.csv 文件，用于记录采样结果
    csv_sampled_path = "/mnt/merged_nvme/lht/TRELLIS/objaverse_sketchfab/importance_sampled.csv"
    if os.path.exists(csv_sampled_path):
        df_sampled = pd.read_csv(csv_sampled_path)
    else:
        df_sampled = pd.DataFrame(columns=[
            "sha256",
            "importance_sample_path",
            "importance_sample_count"
        ])

    # 根据 df_sampled 里已有的 sha256, 进行排重
    existing_sha256s = set(df_sampled["sha256"].unique())
    df_unsampled = df_meta[~df_meta["sha256"].isin(existing_sha256s)].copy()
    if df_unsampled.empty:
        print("No unsampled models found. Nothing to do.")
        return

    print(f"Found {len(df_unsampled)} unsampled models to process.\n", flush=True)
    rows = list(df_unsampled.to_dict("records"))

    max_workers = 16
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务
        for row in rows:
            future = executor.submit(process_one_model, row)
            futures.append(future)

        # 进度条 & 批量写回 CSV
        partial_buffer = []
        count_since_last_write = 0
        with tqdm(total=len(rows), desc="IMPORTANCE SAMPLING", dynamic_ncols=True) as progress_bar:
            for future in concurrent.futures.as_completed(futures):
                result_dict = future.result()
                partial_buffer.append(result_dict)
                count_since_last_write += 1
                progress_bar.update(1)

                # 每处理100个就写回一次
                if count_since_last_write >= 20:
                    new_df = pd.DataFrame(partial_buffer)
                    partial_buffer.clear()
                    df_sampled = pd.concat([df_sampled, new_df], ignore_index=True)
                    df_sampled.to_csv(csv_sampled_path, index=False)
                    count_since_last_write = 0

        # 收尾：剩余的还没写入
        if partial_buffer:
            new_df = pd.DataFrame(partial_buffer)
            df_sampled = pd.concat([df_sampled, new_df], ignore_index=True)
            df_sampled.to_csv(csv_sampled_path, index=False)

    print("\nAll importance sampling done.")
    print(f"Updated sampling info saved to {csv_sampled_path}", flush=True)


if __name__ == "__main__":
    main()
