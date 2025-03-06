import pandas as pd
import numpy as np


def add_train_val_test_split(input_csv, output_csv):
    # 读取输入的 CSV 文件
    df = pd.read_csv(input_csv)

    # 定义类别及对应的概率
    choices = ['train', 'val', 'test']
    probabilities = [0.93, 0.02, 0.05]

    # 为每一行随机分配一个类别
    df['split'] = np.random.choice(choices, size=len(df), p=probabilities)

    # 保存到新的 CSV 文件
    df.to_csv(output_csv, index=False)
    print(f"Updated CSV saved to {output_csv}")


if __name__ == "__main__":
    input_csv = "/data/3D_Dataset/TRELLIS/objaverse_sketchfab/metadata.csv"  # 修改为你的 CSV 文件路径
    output_csv = "/data/3D_Dataset/TRELLIS/objaverse_sketchfab/metadata.csv"  # 修改为你希望保存的文件路径
    add_train_val_test_split(input_csv, output_csv)