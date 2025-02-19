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
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
###############################################################################
# 1. 单个模型的处理函数（不变）
###############################################################################

def load_ply_points_normals(ply_path):
    """ 使用Open3D读取PLY文件中的点和法线 """
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)  # shape: (N, 3)
    normals = np.asarray(pcd.normals)  # shape: (N, 3)
    return points, normals

def world2camera_extrinsic(transform_matrix):
    """ 在NeRF-like数据集中，transform_matrix 通常是 "camera to world" """
    tm = torch.from_numpy(transform_matrix).float()
    extrinsic = torch.linalg.inv(tm)
    return extrinsic

def compute_intrinsics_from_fov(camera_angle_x, width, height):
    """ 根据FOV和图像尺寸计算相机内参 """
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
    """ 将 points 从原坐标系转换到渲染使用的坐标系 """
    offset = np.array(offset, dtype=np.float32).reshape(1, 3)
    new_points = points * scale + offset
    return new_points

# def compute_uv_coordinates_and_weights(points_w, normals_w, extrinsics, intrinsics, image_width, image_height):
#     """ 计算 3D 点的 UV 坐标（归一化到 [-1,1]）及与相机方向的余弦权重 """
#     N = points_w.shape[0]
#     points_h = np.hstack([points_w, np.ones((N, 1))])  # (N, 4)
#     points_h = torch.tensor(points_h, dtype=torch.float32)
#
#     normals_w = torch.tensor(normals_w, dtype=torch.float32)
#
#     # world->camera
#     cam_coords = points_h @ extrinsics.T  # [N,4]
#     cam_xyz = cam_coords[:, :3] / (cam_coords[:, 3:] + 1e-8)  # (N,3)
#
#     # camera coords -> pixel coords
#     proj_2d = cam_xyz @ intrinsics.T  # (N,3)
#     eps = 1e-8
#     x_pix = proj_2d[:, 0] / (proj_2d[:, 2] + eps)
#     y_pix = proj_2d[:, 1] / (proj_2d[:, 2] + eps)
#
#     u_norm = (x_pix / image_width) * 2 - 1
#     v_norm = (y_pix / image_height) * 2 - 1
#     uv = torch.stack([u_norm, v_norm], dim=-1)  # (N,2)
#
#     # 法线与相机前向方向的余弦(假设+Z是前向)
#     camera_dir_cam = torch.tensor([0, 0, 1], dtype=torch.float32)
#     R = extrinsics[:3, :3]
#     normals_cam = normals_w @ R.T
#     cos_theta = torch.sum(normals_cam * camera_dir_cam, dim=-1)
#     weights = torch.clamp(cos_theta, min=0.0)
#
#     return uv, weights

def compute_uv_coordinates_and_weights(points_w, normals_w, extrinsics, intrinsics, image_width, image_height):
    """ 计算 3D 点的 UV 坐标（归一化到 [-1,1]）及与相机方向的余弦权重 """
    N = points_w.shape[0]
    # 转换为齐次坐标，并转换为 torch tensor
    points_h = np.hstack([points_w, np.ones((N, 1))])  # (N, 4)
    points_h = torch.tensor(points_h, dtype=torch.float32)
    normals_w = torch.tensor(normals_w, dtype=torch.float32)

    # world -> camera 坐标转换
    cam_coords = points_h @ extrinsics.T  # (N, 4)
    cam_xyz = cam_coords[:, :3] / (cam_coords[:, 3:] + 1e-8)  # (N, 3)

    # camera coords -> 像素坐标
    proj_2d = cam_xyz @ intrinsics.T  # (N, 3)
    eps = 1e-8
    x_pix = proj_2d[:, 0] / (proj_2d[:, 2] + eps)
    y_pix = proj_2d[:, 1] / (proj_2d[:, 2] + eps)

    # 根据正确代码的归一化方式，先减去图像中心，再除以半宽高
    u_norm = (x_pix - image_width / 2) / (image_width / 2)
    v_norm = (y_pix - image_height / 2) / (image_height / 2)
    uv = torch.stack([u_norm, v_norm], dim=-1)  # (N, 2)

    # 计算法线与相机前向（假设+Z为前向）方向的余弦权重
    camera_dir_cam = torch.tensor([0, 0, 1], dtype=torch.float32)
    R = extrinsics[:3, :3]
    normals_cam = normals_w @ R.T
    cos_theta = torch.sum(normals_cam * camera_dir_cam, dim=-1)
    weights = torch.clamp(cos_theta, min=0.0)

    return uv, weights

def weighted_average_features(sampled_features, weight):
    """ 对多个视角的特征做加权平均 """
    weight = weight.unsqueeze(-1)  # [n_views, N, 1]
    weighted_features = sampled_features * weight
    summed_features = weighted_features.sum(dim=0)  # [N, dim]
    summed_weights = weight.sum(dim=0)  # [N, 1]

    summed_weights = summed_weights + 1e-8
    averaged_features = summed_features / summed_weights
    return averaged_features

def process_and_extract_features(row, model, preprocess, base_path, device, num_samples=2048):
    sha = row["sha256"]
    ply_path = row["poisson_disk_sampled_pointcloud_path"]

    if not os.path.exists(ply_path):
        return {"sha256": sha, "status": "PLY_not_found"}

    # pcd = load_ply_points_normals(ply_path)

    pcd = o3d.io.read_point_cloud(ply_path)
    pcd_downsampled = pcd.farthest_point_down_sample(num_samples)

    # 保存 FPS 下采样后的点云文件
    out_point_fps_dir = "/mnt/merged_nvme/lht/TRELLIS/objaverse_sketchfab/fps_downsample"
    os.makedirs(out_point_fps_dir, exist_ok=True)
    fps_out_path = os.path.join(out_point_fps_dir, f"{sha}_fps.ply")
    o3d.io.write_point_cloud(fps_out_path, pcd_downsampled)

    ### !!! 坐标系变换 !!!
    R = np.array([[1, 0, 0],  # x轴不变
                  [0, 0, -1],  # y轴变为-z轴
                  [0, 1, 0]]) # z轴变为y轴
    ### !!!!!!!!!!!!!!!!!

    points = np.asarray(pcd_downsampled.points) @ R.T  # 提取下采样后的点云
    normals = np.asarray(pcd_downsampled.normals) @ R.T  # 提取法向量

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

    all_sampled_feats = []
    all_weights = []
    all_uv = []

    image_width = 518
    image_height = 518

    # 创建一个用于批量推理的图像列表
    img_batch = []

    # 将所有视角的图像堆叠到一个batch
    for i, frame in enumerate(frames):
        png_name = frame["file_path"]
        img_path = os.path.join(render_dir, png_name)
        if not os.path.exists(img_path):
            continue

        M_cam2world = np.array(frame["transform_matrix"], dtype=np.float32)
        extrinsics = world2camera_extrinsic(M_cam2world)

        camera_angle_x = frame["camera_angle_x"]
        intrinsics = compute_intrinsics_from_fov(camera_angle_x, image_width, image_height)

        img = Image.open(img_path).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0).to(device)  # [1,3,H,W]
        img_batch.append(img_tensor)

    # 将所有图像堆叠成一个batch
    if len(img_batch) == 0:
        return {"sha256": sha, "status": "no_valid_view_images"}

    img_batch = torch.cat(img_batch, dim=0)  # 形成一个 [n_views, 3, H, W] 的 batch

    # 执行一次 batch 推理
    with torch.no_grad():
        outputs = model(img_batch)  # 将整个 batch 传入模型进行推理
        last_hidden_states = outputs.last_hidden_state  # [n_views, tokens, C]

    grid_size = image_width // 14
    feat_map = last_hidden_states[:, 5:, :]
    feat_map = feat_map
    B, HW, C = feat_map.shape
    feat_map = feat_map.permute(0, 2, 1).reshape(B, C, grid_size, grid_size)

    # 计算 UV 和权重
    for i, frame in enumerate(frames):
        png_name = frame["file_path"]
        img_path = os.path.join(render_dir, png_name)
        if not os.path.exists(img_path):
            continue

        M_cam2world = np.array(frame["transform_matrix"], dtype=np.float32)
        extrinsics = world2camera_extrinsic(M_cam2world)

        camera_angle_x = frame["camera_angle_x"]
        intrinsics = compute_intrinsics_from_fov(camera_angle_x, image_width, image_height)

        uv, w = compute_uv_coordinates_and_weights(
            points_w, normals_w, extrinsics, intrinsics,
            image_width, image_height
        )

        uv_4d = uv.unsqueeze(0)  # [1,1,N,2]
        all_uv.append(uv_4d)
        all_weights.append(w)
    all_uv = torch.stack(all_uv).to(device)
    sampled = F.grid_sample(
        feat_map,  # [n_views, 1024, grid_size, grid_size]
        all_uv,    # [n_views, 1, N, 2]
        mode='bilinear',
        align_corners=False,
    ).squeeze(2).permute(0, 2, 1)
    # print(sampled)
    weights_tensor = torch.stack(all_weights, dim=0).to(device)  # [n_views, N]
    final_features = weighted_average_features(sampled, weights_tensor)

    out_dir = "/mnt/merged_nvme/lht/TRELLIS/objaverse_sketchfab/features"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{sha}.safetensors")
    # torch.save(final_features, out_path)
    save_file({"final_features": final_features.cpu()}, out_path)

    return {
        "sha256": sha,
        "fps_downsample_path": fps_out_path,
        "feature_path": out_path
    }

###############################################################################
# 2. 使用 DataParallel 的多GPU推理主函数
###############################################################################

def main():
    base_path = "/mnt/merged_nvme/lht/TRELLIS/objaverse_sketchfab"
    csv_metadata_path = os.path.join(base_path, "metadata.csv")
    df_meta = pd.read_csv(csv_metadata_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # rank = int(os.environ["RANK"])
    # print(rank)
    # device = torch.device(f"cuda:{rank}")
    # torch.distributed.init_process_group("nccl", device_id=device)

    model = AutoModel.from_pretrained('facebook/dinov2-with-registers-giant').to(device).eval()

    # 使用 DataParallel 进行多GPU推理
    # model = torch.nn.DataParallel(model)  # 自动将模型复制到多个GPU


    model = model.to(device)

    image_width = 518
    image_height = 518
    preprocess = transforms.Compose([
        transforms.Resize((image_height, image_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    csv_features_path = os.path.join(base_path, "features.csv")
    if os.path.exists(csv_features_path):
        df_features = pd.read_csv(csv_features_path)
    else:
        df_features = pd.DataFrame(columns=["sha256", "fps_downsample_path", "feature_path"])

    processed_sha256s = set(df_features["sha256"].unique())
    df_todo = df_meta[~df_meta["sha256"].isin(processed_sha256s)].copy()
    if df_todo.empty:
        print("All models in metadata.csv have been processed. Nothing to do.")
        return

    rows = df_todo.to_dict("records")

    max_workers = 4
    partial_buffer = []
    count_since_last_write = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for row in rows:
            future = executor.submit(process_and_extract_features, row, model, preprocess, base_path, device)
            futures.append(future)

        with tqdm(total=len(rows), desc="Extracting DINOv2 features", ncols=100) as pbar:
            for future in concurrent.futures.as_completed(futures):
                result_dict = future.result()
                partial_buffer.append(result_dict)
                count_since_last_write += 1

                pbar.update(1)

                if count_since_last_write >= 100:
                    tmp_df = pd.DataFrame(partial_buffer)
                    partial_buffer.clear()
                    df_features = pd.concat([df_features, tmp_df], ignore_index=True)
                    df_features.to_csv(csv_features_path, index=False)
                    count_since_last_write = 0

    if partial_buffer:
        tmp_df = pd.DataFrame(partial_buffer)
        df_features = pd.concat([df_features, tmp_df], ignore_index=True)
        df_features.to_csv(csv_features_path, index=False)

    print("\nAll feature extraction done!")
    print(f"Results saved/updated to {csv_features_path}")

if __name__ == "__main__":
    main()
