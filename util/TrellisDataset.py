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
from safetensors.torch import load_file
from extensions.FlexiCubes.examples.util import *

# import h5py

class TrellisDataset(data.Dataset):
    def __init__(self, split, base_path = "/mnt/merged_nvme/lht/TRELLIS/objaverse_sketchfab", categories=['03001627'], transform=None, sampling=True, num_samples=4096,
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
        print(f"This is {self.split} Dataset.")
        for _, row in self.df_meta.iterrows():
            if row['split'] == self.split:
                self.models.append({
                    'sha256': row['sha256'],
                    'feature_path': row['feature_path'],
                    'fps_downsample_path': row['fps_downsample_path']
                })
        print(f"Totally {len(self.models)} models in {self.split} Dataset.")
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
        mesh_path = os.path.join(self.base_path, self.models[idx]['local_path'])

        gt_mesh = load_mesh(mesh_path, 'cuda')

        # image = None

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

        data = load_file(feature_path)
        feat = data['final_features']
        if self.transform:
            surface = self.transform(surface)

        if self.return_surface:
            return surface, gt_mesh, feat

        else:
            return gt_mesh, feat



    def __len__(self):
        if self.split != 'train':
            return len(self.models)
        else:
            return len(self.models) * self.replica
