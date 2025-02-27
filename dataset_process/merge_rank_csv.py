#!/usr/bin/env python3
import os
import glob
import pandas as pd


def main():
    # 假设你的这些 CSV 都在这个目录下
    base_path = "/data/3D_Dataset/TRELLIS/objaverse_sketchfab"
    final_csv_name = "features.csv"

    # 拿到现有的 features_{rank}.csv 文件列表
    pattern = os.path.join(base_path, "features_*.csv")
    rank_csvs = glob.glob(pattern)

    if not rank_csvs:
        print("No features_{rank}.csv files found. Nothing to merge.")
        return

    # 先把所有 features_{rank}.csv 读进来合并
    df_list = []
    for csv_file in rank_csvs:
        print(f"Reading {csv_file}")
        df_list.append(pd.read_csv(csv_file))
    df_merged = pd.concat(df_list, ignore_index=True)

    # 再看原先是否已有 features.csv，把它也读进来
    final_csv_path = os.path.join(base_path, final_csv_name)
    if os.path.exists(final_csv_path):
        print(f"Reading existing {final_csv_path}")
        df_existing = pd.read_csv(final_csv_path)
        # 拼接后再去重
        df_merged = pd.concat([df_existing, df_merged], ignore_index=True)

    # 根据 sha256 去重，保留最后一次出现的记录
    df_merged.drop_duplicates(subset=["sha256"], keep="last", inplace=True)

    # 写回到同一个 features.csv
    df_merged.to_csv(final_csv_path, index=False)
    print(f"Merged CSV saved to {final_csv_path}")

    # 如果希望合并后删除分片文件，可以取消注释下面这段
    # for csv_file in rank_csvs:
    #     os.remove(csv_file)
    #     print(f"Removed {csv_file}")


if __name__ == "__main__":
    main()
