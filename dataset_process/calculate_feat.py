import os
import json
import pandas as pd
import numpy as np
import torch
import open3d as o3d
from PIL import Image
from torchvision import transforms
from transformers import AutoModel
from torch.functional import F
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from safetensors.torch import save_file

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

###############################################################################
# 1. 单个模型的处理函数（与你之前的保持一致，仅在这里简化展示）
###############################################################################

def load_ply_points_normals(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    return points, normals

def world2camera_extrinsic(transform_matrix):
    tm = torch.from_numpy(transform_matrix).float()
    extrinsic = torch.linalg.inv(tm)
    return extrinsic

def compute_intrinsics_from_fov(camera_angle_x, width, height):
    f = (width / 2.0) / np.tan(camera_angle_x / 2.0)
    cx = width / 2.0
    cy = height / 2.0
    intrinsics = torch.tensor([
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1.0]
    ], dtype=torch.float32)
    return intrinsics

def apply_scale_offset(points, scale, offset):
    offset = np.array(offset, dtype=np.float32).reshape(1, 3)
    new_points = points * scale + offset
    return new_points

def compute_uv_coordinates_and_weights(points_w, normals_w, extrinsics, intrinsics, image_width, image_height):
    N = points_w.shape[0]
    points_h = np.hstack([points_w, np.ones((N, 1))])
    points_h = torch.tensor(points_h, dtype=torch.float32)
    normals_w = torch.tensor(normals_w, dtype=torch.float32)

    # world->camera
    cam_coords = points_h @ extrinsics.T
    cam_xyz = cam_coords[:, :3] / (cam_coords[:, 3:] + 1e-8)

    # camera coords->像素坐标
    proj_2d = cam_xyz @ intrinsics.T
    eps = 1e-8
    x_pix = proj_2d[:, 0] / (proj_2d[:, 2] + eps)
    y_pix = proj_2d[:, 1] / (proj_2d[:, 2] + eps)

    u_norm = (x_pix - image_width / 2) / (image_width / 2)
    v_norm = (y_pix - image_height / 2) / (image_height / 2)
    uv = torch.stack([u_norm, v_norm], dim=-1)

    camera_dir_cam = torch.tensor([0, 0, 1], dtype=torch.float32)
    R = extrinsics[:3, :3]
    normals_cam = normals_w @ R.T
    cos_theta = torch.sum(normals_cam * camera_dir_cam, dim=-1)
    weights = torch.clamp(cos_theta, min=0.0)

    return uv, weights

def weighted_average_features(sampled_features, weight):
    weight = weight.unsqueeze(-1)
    weighted_features = sampled_features * weight
    summed_features = weighted_features.sum(dim=0)
    summed_weights = weight.sum(dim=0) + 1e-8
    averaged_features = summed_features / summed_weights
    return averaged_features

def process_and_extract_features(row, model, preprocess, base_path, device, num_samples=2048):
    sha = row["sha256"]
    ply_path = os.path.join(base_path, "poisson_disk_samples", f"{sha}.ply")
    if not os.path.exists(ply_path):
        return {"sha256": sha, "status": "PLY_not_found"}

    # 下采样点云
    pcd = o3d.io.read_point_cloud(ply_path)
    pcd_downsampled = pcd.farthest_point_down_sample(num_samples)

    # 保存 FPS 下采样后的点云文件
    out_point_fps_dir = os.path.join(base_path, "fps_downsample")
    os.makedirs(out_point_fps_dir, exist_ok=True)
    fps_out_path = os.path.join(out_point_fps_dir, f"{sha}_fps.ply")
    o3d.io.write_point_cloud(fps_out_path, pcd_downsampled)

    # 额外的坐标系变换
    R = np.array([[1, 0, 0],
                  [0, 0, -1],
                  [0, 1, 0]])
    points = np.asarray(pcd_downsampled.points) @ R.T
    normals = np.asarray(pcd_downsampled.normals) @ R.T

    render_dir = os.path.join(base_path, "renders", sha)
    tfm_path = os.path.join(render_dir, "transforms.json")
    if not os.path.exists(tfm_path):
        return {"sha256": sha, "status": "transforms_not_found"}

    with open(tfm_path, "r") as f:
        tfm_data = json.load(f)

    scale = tfm_data["scale"]
    offset = tfm_data["offset"]
    points_w = apply_scale_offset(points, scale, offset)
    normals_w = normals

    frames = tfm_data["frames"]
    n_views = len(frames)
    if n_views == 0:
        return {"sha256": sha, "status": "no_views_in_transforms"}

    image_width = 518
    image_height = 518

    # 先批量处理所有图像
    img_batch = []
    valid_extrinsics = []
    valid_intrinsics = []

    for frame in frames:
        png_name = frame["file_path"]
        img_path = os.path.join(render_dir, png_name)
        if not os.path.exists(img_path):
            continue
        M_cam2world = np.array(frame["transform_matrix"], dtype=np.float32)
        extrinsics = world2camera_extrinsic(M_cam2world)

        camera_angle_x = frame["camera_angle_x"]
        intrinsics = compute_intrinsics_from_fov(camera_angle_x, image_width, image_height)

        img = Image.open(img_path).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        img_batch.append(img_tensor)

        valid_extrinsics.append(extrinsics)
        valid_intrinsics.append(intrinsics)

    if len(img_batch) == 0:
        return {"sha256": sha, "status": "no_valid_view_images"}

    img_batch = torch.cat(img_batch, dim=0).to(device)  # [n_views,3,H,W]
    with torch.no_grad():
        outputs = model(img_batch)
        last_hidden_states = outputs.last_hidden_state  # [n_views, tokens, dim]

    # 假设特征图尺寸 / patch_size 已经固定，这里仅做演示
    grid_size = image_width // 14
    feat_map = last_hidden_states[:, 5:, :]  # 略去前面一些token等操作只是示例
    B, HW, C = feat_map.shape
    feat_map = feat_map.permute(0, 2, 1).reshape(B, C, grid_size, grid_size)

    all_uv = []
    all_weights = []
    for i in range(len(valid_extrinsics)):
        uv, w = compute_uv_coordinates_and_weights(
            points_w, normals_w,
            valid_extrinsics[i], valid_intrinsics[i],
            image_width, image_height
        )
        uv_4d = uv.unsqueeze(0).unsqueeze(0).to(device)  # [1,1,N,2]
        all_uv.append(uv_4d)
        all_weights.append(w)

    all_uv = torch.cat(all_uv, dim=0)  # [n_views,1,N,2]
    sampled = F.grid_sample(feat_map, all_uv, mode='bilinear', align_corners=False)
    sampled = sampled.squeeze(2).permute(0, 2, 1)  # [n_views,N,dim]

    weights_tensor = torch.stack(all_weights, dim=0).to(device)  # [n_views, N]
    final_features = weighted_average_features(sampled, weights_tensor)

    out_dir = os.path.join(base_path, "features")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{sha}.safetensors")
    save_file({"final_features": final_features.cpu()}, out_path)

    return {
        "sha256": sha,
        "fps_downsample_path": fps_out_path,
        "feature_path": out_path,
        "status": "done"
    }

###############################################################################
# 2. 主函数：基于 rank/world_size 进行任务切分，并写到 features_{rank}.csv
###############################################################################

# cmd: HF_ENDPOINT=https://hf-mirror.com RANK=0 WORLD_SIZE=8 python calculate_feat.py
def main():
    base_path = "/data/3D_Dataset/TRELLIS/objaverse_sketchfab"
    csv_metadata_path = os.path.join(base_path, "metadata.csv")
    df_meta = pd.read_csv(csv_metadata_path)

    # 读取/判断分布式环境变量
    # 如果没有设置，则默认为单卡(single-rank)运行
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained('facebook/dinov2-with-registers-giant').to(device).eval()

    image_width = 518
    image_height = 518
    preprocess = transforms.Compose([
        transforms.Resize((image_height, image_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # 维护一个总的 features.csv（如果它已经存在的话）
    csv_features_path = os.path.join(base_path, "features.csv")
    if os.path.exists(csv_features_path):
        df_features = pd.read_csv(csv_features_path)
    else:
        df_features = pd.DataFrame(columns=["sha256", "fps_downsample_path", "feature_path", "status"])

    # 找到尚未处理的记录
    processed_sha256s = set(df_features["sha256"].unique())
    df_todo = df_meta[~df_meta["sha256"].isin(processed_sha256s)].copy()
    if df_todo.empty:
        print(f"[Rank {rank}] All models in metadata.csv have been processed. Nothing to do.")
        return

    # 把任务分配给各个 rank
    # 简单的做法：对 todolist 做切片
    df_todo_rank = df_todo.iloc[rank::world_size]  # 每隔 world_size 一条分给一个 rank
    if df_todo_rank.empty:
        print(f"[Rank {rank}] No tasks assigned after slicing (len(df_todo)={len(df_todo)}). Done.")
        return

    rows = df_todo_rank.to_dict("records")
    print(f"[Rank {rank}] Number of tasks to process: {len(rows)}")

    # 每个 rank 的输出文件
    # 如果你希望每个 rank 都写自己的文件，后续再合并，可以这样做
    rank_csv_path = os.path.join(base_path, f"features_{rank}.csv")

    max_workers = 1
    partial_buffer = []
    count_since_last_write = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for row in rows:
            future = executor.submit(process_and_extract_features, row, model, preprocess, base_path, device)
            futures.append(future)

        with tqdm(total=len(rows), desc=f"Rank {rank} extracting", ncols=100) as pbar:
            for future in concurrent.futures.as_completed(futures):
                result_dict = future.result()
                partial_buffer.append(result_dict)
                count_since_last_write += 1
                pbar.update(1)

                # 每隔 100 条写一次 CSV
                if count_since_last_write >= 100:
                    tmp_df = pd.DataFrame(partial_buffer)
                    partial_buffer.clear()
                    if os.path.exists(rank_csv_path):
                        df_existing_rank = pd.read_csv(rank_csv_path)
                        df_existing_rank = pd.concat([df_existing_rank, tmp_df], ignore_index=True)
                        df_existing_rank.to_csv(rank_csv_path, index=False)
                    else:
                        tmp_df.to_csv(rank_csv_path, index=False)
                    count_since_last_write = 0

    # 把剩余没写完的也写一次
    if partial_buffer:
        tmp_df = pd.DataFrame(partial_buffer)
        if os.path.exists(rank_csv_path):
            df_existing_rank = pd.read_csv(rank_csv_path)
            df_existing_rank = pd.concat([df_existing_rank, tmp_df], ignore_index=True)
            df_existing_rank.to_csv(rank_csv_path, index=False)
        else:
            tmp_df.to_csv(rank_csv_path, index=False)

    print(f"[Rank {rank}] Done! Results saved to {rank_csv_path}")

if __name__ == "__main__":
    main()
