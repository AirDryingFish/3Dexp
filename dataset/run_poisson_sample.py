import concurrent.futures
import pandas as pd
import open3d as o3d
import numpy as np
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

# 1) 屏蔽 Open3D 的日志(如Poisson采样输出)
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


def load_triangle_mesh_with_open3d(mesh_file):
    """
    用 Open3D 加载一个三角网格文件(如 .glb/.obj/.stl等)。
    如果需要可以额外处理(如法线计算)。
    """
    mesh = o3d.io.read_triangle_mesh(mesh_file, print_progress=False)
    if not mesh or not mesh.has_triangles():
        print(f"Warning: {mesh_file} has no triangles or failed to load.")
        return None
    return mesh


def poisson_disk_sampling(mesh_file, out_file, number_of_points=50000, init_factor=5):
    """
    对 mesh_file 执行 Poisson Disk 采样，并将结果存储到 out_file。
    返回保存的点的数量(或其他信息)。
    """
    mesh = load_triangle_mesh_with_open3d(mesh_file)
    if mesh is None:
        print(f"Skip {mesh_file} because mesh is invalid.")
        return 0

    pcd = mesh.sample_points_poisson_disk(
        number_of_points=number_of_points,
        init_factor=init_factor,
        use_triangle_normal=False
    )
    pts_array = np.asarray(pcd.points)
    # 这里可打印采样结果,若想屏蔽输出就注释掉:
    # print(f"[{mesh_file}] Poisson Disk sampling done: {pts_array.shape[0]} points.")

    # 保存结果
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    o3d.io.write_point_cloud(out_file, pcd)
    return pts_array.shape[0]


def process_one_model(row):
    """
    针对 metadata.csv 的一行(row)，执行采样并返回:
      {
        "sha256": sha256,
        "poisson_disk_sampled_pointcloud_path": 输出路径,
        "poisson_disk_sampled_pointcloud_count": 采样后的点数
      }
    """
    base_path = "/mnt/merged_nvme/lht/TRELLIS/objaverse_sketchfab/"
    mesh_file = os.path.join(base_path, row["local_path"])
    sha256_id = row["sha256"]

    # 自定义保存路径
    out_file = f"/mnt/merged_nvme/lht/TRELLIS/objaverse_sketchfab/poisson_disk_samples/{sha256_id}.ply"

    result_dict = {
        "sha256": sha256_id,
        "poisson_disk_sampled_pointcloud_path": "",
        "poisson_disk_sampled_pointcloud_count": 0
    }
    try:
        sampled_count = poisson_disk_sampling(
            mesh_file=mesh_file,
            out_file=out_file,
            number_of_points=50000,
            init_factor=1
        )
        result_dict["poisson_disk_sampled_pointcloud_path"] = out_file
        result_dict["poisson_disk_sampled_pointcloud_count"] = sampled_count
    except Exception as e:
        print(f"Error processing {mesh_file}: {e}", flush=True)
        # 保持 result_dict 中 path=count=0

    return result_dict


def main():
    # 原始的metadata文件,包含sha256和local_path等
    csv_metadata_path = "/mnt/merged_nvme/lht/TRELLIS/objaverse_sketchfab/metadata.csv"
    df_meta = pd.read_csv(csv_metadata_path)

    # 确保 metadata.csv 至少包含 "sha256", "local_path"
    if "sha256" not in df_meta.columns or "local_path" not in df_meta.columns:
        print("metadata.csv must contain 'sha256' and 'local_path' columns!")
        return

    # 读取或创建 poisson_sampled.csv 文件, 用于记录采样结果
    csv_sampled_path = "/mnt/merged_nvme/lht/TRELLIS/objaverse_sketchfab/poisson_sampled.csv"
    if os.path.exists(csv_sampled_path):
        df_sampled = pd.read_csv(csv_sampled_path)
    else:
        df_sampled = pd.DataFrame(columns=[
            "sha256",
            "poisson_disk_sampled_pointcloud_path",
            "poisson_disk_sampled_pointcloud_count"
        ])

    # 根据 df_sampled 里已有的 sha256, 跳过
    existing_sha256s = set(df_sampled["sha256"].unique())
    df_unsampled = df_meta[~df_meta["sha256"].isin(existing_sha256s)].copy()
    if df_unsampled.empty:
        print("No unsampled models found. Nothing to do.")
        return

    print(f"Found {len(df_unsampled)} unsampled models to process.\n", flush=True)

    rows = list(df_unsampled.to_dict("records"))

    max_workers = 8
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务
        for row in rows:
            future = executor.submit(process_one_model, row)
            futures.append(future)

        # 用 as_completed + tqdm 显示进度
        partial_buffer = []        # 用于暂存 result_dict
        count_since_last_write = 0 # 计数器
        with tqdm(total=len(rows), desc="POISSON DISK SAMPLING", dynamic_ncols=True) as progress_bar:
            for future in concurrent.futures.as_completed(futures):
                # 拿到结果
                result_dict = future.result()
                partial_buffer.append(result_dict)
                count_since_last_write += 1

                progress_bar.update(1)

                # 每满10条, 写一次 CSV
                if count_since_last_write >= 100:
                    new_df = pd.DataFrame(partial_buffer)
                    partial_buffer.clear()  # 清空buffer
                    df_sampled = pd.concat([df_sampled, new_df], ignore_index=True)
                    df_sampled.to_csv(csv_sampled_path, index=False)
                    count_since_last_write = 0

        # 全部完成后, 若buffer里还剩下零散条目 => 再写一次
        if partial_buffer:
            new_df = pd.DataFrame(partial_buffer)
            df_sampled = pd.concat([df_sampled, new_df], ignore_index=True)
            df_sampled.to_csv(csv_sampled_path, index=False)

    print("\nAll sampling done.")
    print(f"Updated sampling info saved to {csv_sampled_path}", flush=True)


if __name__ == "__main__":
    main()
