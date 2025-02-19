import os
import glob
import random

import yaml

import torch
from torch.utils import data
from torchvision import transforms

import open3d as o3d
import numpy as np
import pandas as pd
from PIL import Image

# import h5py

class TrellisDataset(data.Dataset):
    def __init__(self, dataset_folder, split, base_path = "/mnt/merged_nvme/lht/TRELLIS/objaverse_sketchfab", categories=['03001627'], transform=None, sampling=True, num_samples=4096,
                 return_surface=True, surface_sampling=True, pc_size=2048, replica=16, image_folder=None):

        self.pc_size = pc_size

        self.transform = transform
        self.num_samples = num_samples
        self.sampling = sampling
        self.split = split

        self.return_surface = return_surface
        self.surface_sampling = surface_sampling

        self.base_path = base_path
        # self.dataset_folder = dataset_folder
        # self.point_folder = os.path.join(self.dataset_folder, 'ShapeNetV2_point')
        # self.mesh_folder = os.path.join(self.dataset_folder, 'ShapeNetV2_watertight')

        self.metadata_path = os.path.join(self.base_path, 'metadata.csv')
        self.df_meta = pd.read_csv(self.metadata_path)
        # self.point_path = self.df_meta
        self.models = []
        for _, row in self.df_meta.iterrows():
            self.models.append({
                'sha256': row['sha256'],
                'feature_path': row['feature_path'],
                'fps_downsample_path': row['fps_downsample_path']
            })

        # self.image_folder = image_folder
        # self.image_cond_transform = transforms.ToTensor()

        # self.models = []
        # for c_idx, c in enumerate(categories):
        #     subpath = os.path.join(self.point_folder, c)
        #     assert os.path.isdir(subpath)
        #
        #     split_file = os.path.join(subpath, split + '.lst')
        #     with open(split_file, 'r') as f:
        #         models_c = f.read().split('\n')
        #
        #     self.models += [
        #         {'category': c, 'model': m.replace('.npz', '')}
        #         for m in models_c
        #     ]

        self.replica = replica

    def __getitem__(self, idx):
        idx = idx % len(self.models)

        # category = self.models[idx]['category']
        # model = self.models[idx]['model']

        fps_downsample_path = self.models[idx]['fps_downsample_path']
        feature_path = self.models[idx]['feature_path']
        label_path = self.models[idx]['label_path']

        image = None

        # if self.image_folder != None:
        #     image_renders_path = os.path.join(self.image_folder, category, model + '_dino.pt')
        #     # image = Image.open(image_renders_path)
        #     # image = self.image_cond_transform(image)
        #     image_emb = torch.load(image_renders_path)

        try:
            with np.load(label_path) as data:
                # needed: {ndarray}
                queries_label = data['queries_label']
        except Exception as e:
            print(e)
            print(label_path)

        if self.return_surface:
            # surface = data['points'].astype(np.float32)  # [100000, 3]
            # [4096, 3]
            data = o3d.io.read_point_cloud(fps_downsample_path)
            surface = np.asarray(data.points)
            # surface = surface * scale

            if self.surface_sampling:
                # pc_size: 采样后的点云数量 (default: 2048)
                ind = np.random.default_rng().choice(surface.shape[0], self.pc_size, replace=False)
                surface = surface[ind]
            surface = torch.from_numpy(surface)

        # if self.sampling:
        #     ind = np.random.default_rng().choice(vol_points.shape[0], self.num_samples, replace=False)
        #     vol_points = vol_points[ind]
        #     vol_label = vol_label[ind]
        #
        #     ind = np.random.default_rng().choice(near_points.shape[0], self.num_samples, replace=False)
        #     near_points = near_points[ind]
        #     near_label = near_label[ind]

        queries_label = torch.from_numpy(queries_label).float()

        labels = queries_label
        density = 128
        x_coor = np.linspace(-0.5, 0.5, density + 1)  # [128]^3的网格，每个坐标上有129个点
        y_coor = np.linspace(-0.5, 0.5, density + 1)
        z_coor = np.linspace(-0.5, 0.5, density + 1)

        xv, yv, zv = np.meshgrid(x_coor, y_coor, z_coor)
        # [3, 129, 129, 129] -> [3, 129 * 129 * 129] -> [129 * 129 * 129, 3]
        points = torch.from_numpy(np.stack([xv, yv, zv]).astype(np.float32)).view(3, -1).transpose(0, 1)

        feat = torch.load(feature_path)
        if self.transform:
            surface, points = self.transform(surface, points)

        if self.return_surface:
            return points, labels, surface, feat

        else:
            return points, labels, feat

    def __len__(self):
        if self.split != 'train':
            return len(self.models)
        else:
            return len(self.models) * self.replica

