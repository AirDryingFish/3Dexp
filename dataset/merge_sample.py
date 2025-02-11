import pandas as pd
import numpy as np
import os

def merge_poisson_to_metadata():
    metadata_csv = "/mnt/merged_nvme/lht/TRELLIS/objaverse_sketchfab/metadata.csv"
    sampled_csv = "/mnt/merged_nvme/lht/TRELLIS/objaverse_sketchfab/poisson_sampled.csv"
    output_csv = "/mnt/merged_nvme/lht/TRELLIS/objaverse_sketchfab/metadata.csv"

    # 1) 读取 metadata.csv
    df_meta = pd.read_csv(metadata_csv)

    # 确保 metadata.csv 包含 'sha256' 列
    if "sha256" not in df_meta.columns:
        print("Error: metadata.csv must contain 'sha256' column!")
        return

    # 2) 确保 metadata.csv 里有这两列, 若无则先插入
    #    这能保证后面 update 不会报错
    if "poisson_disk_sampled_pointcloud_path" not in df_meta.columns:
        df_meta["poisson_disk_sampled_pointcloud_path"] = np.nan
    if "poisson_disk_sampled_pointcloud_count" not in df_meta.columns:
        df_meta["poisson_disk_sampled_pointcloud_count"] = np.nan

    # 3) 读取 poisson_sampled.csv
    if not os.path.exists(sampled_csv):
        print("Error: poisson_sampled.csv not found!")
        return
    df_sampled = pd.read_csv(sampled_csv)

    # 确保 poisson_sampled.csv 也包含 'sha256' 列
    if "sha256" not in df_sampled.columns:
        print("Error: poisson_sampled.csv must contain 'sha256' column!")
        return

    # 同样, 确保 df_sampled 有这两个要合并的列(若没有则填充)
    for col in ["poisson_disk_sampled_pointcloud_path", "poisson_disk_sampled_pointcloud_count"]:
        if col not in df_sampled.columns:
            df_sampled[col] = np.nan

    # 4) 将二者都以 'sha256' 为索引
    df_meta.set_index("sha256", inplace=True)
    df_sampled.set_index("sha256", inplace=True)

    # 5) 用 pandas 的 update 方法，把 df_sampled 的两列更新进 df_meta
    #    只更新 df_meta 中对应行, df_sampled有值时才写入
    for col in ["poisson_disk_sampled_pointcloud_path", "poisson_disk_sampled_pointcloud_count"]:
        if col in df_sampled.columns:
            df_meta[col].update(df_sampled[col])  # 只在df_sampled中非NaN时才覆盖

    # 6) reset index, 并保存
    df_meta.reset_index(inplace=True)
    df_meta.to_csv(output_csv, index=False)
    print(f"Merged results saved to {output_csv}")

if __name__ == "__main__":
    merge_poisson_to_metadata()
