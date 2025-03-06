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
# from multiprocessing import Manager

# import h5py

class TrellisDataset(data.Dataset):
    def __init__(self, split, base_path = "/data/3D_Dataset/TRELLIS/objaverse_sketchfab/", categories=['03001627'], transform=None, sampling=True, num_samples=4096,
                 return_surface=True, surface_sampling=True, pc_size=2048, replica=16, image_folder=None):

        self.pc_size = pc_size

        self.transform = transform
        self.num_samples = num_samples
        self.sampling = sampling
        self.split = split

        self.return_surface = return_surface
        self.surface_sampling = surface_sampling

        self.base_path = base_path

        self.metadata_path = os.path.join(self.base_path, 'metadata.csv')
        self.df_meta = pd.read_csv(self.metadata_path)
        self.models = []
        # manager = Manager()
        print(f"This is {self.split} Dataset.")
        for _, row in self.df_meta.iterrows():
            if row['split'] == self.split:
                self.models.append({
                    'sha256': row['sha256'],
                    'local_path': row['local_path'],
                    'feature_path': os.path.join(base_path, "features", f"{row['sha256']}.safetensors"),
                    'fps_downsample_path': os.path.join(base_path, "fps_downsample", f"{row['sha256']}_fps.ply")
                })
                # self.model = manager.list(self.models)
        print(f"Totally {len(self.models)} models in {self.split} Dataset.")

        self.replica = replica

    def __getitem__(self, idx):
        idx = idx % len(self.models)
        fps_downsample_path = self.models[idx]['fps_downsample_path']
        feature_path = self.models[idx]['feature_path']
        mesh_path = os.path.join(self.base_path, self.models[idx]['local_path'])
        default_flag = 0
        try:
            vertices, faces = load_mesh(mesh_path, 'cpu')
        except Exception as e:
            print(f"Loading Mesh {mesh_path} Erorr: {e}")
            print(f"Using default mesh {os.path.join(self.base_path, 'raw/hf-objaverse-v1/glbs/000-113/7aca8c05583c48b8a3bfae6d043e331b.glb')}")
            default_flag = 1
            vertices, faces = load_mesh(os.path.join(self.base_path, 'raw/hf-objaverse-v1/glbs/000-113/7aca8c05583c48b8a3bfae6d043e331b.glb'), 'cpu')

        try:
            if self.return_surface:
                # surface = data['points'].astype(np.float32)  # [100000, 3]
                # [4096, 3]
                data = o3d.io.read_point_cloud(fps_downsample_path)
                surface = np.asarray(data.points)
                R = np.array([[1, 0, 0],
                              [0, 0, -1],
                              [0, 1, 0]])
                surface = surface @ R.T
                # surface = surface * scale

                if self.surface_sampling:
                    # pc_size: 采样后的点云数量 (default: 2048)
                    ind = np.random.default_rng().choice(surface.shape[0], self.pc_size, replace=False)
                    surface = surface[ind]
        except Exception as e:
            print(f"Sampling Error: {e}, using default sampling instead: ")
            default_flag = 1

        data = load_file(feature_path)
        feat = data['final_features']

        if default_flag == 1:
            vertices, faces = load_mesh(
                os.path.join(self.base_path, 'raw/hf-objaverse-v1/glbs/000-113/7aca8c05583c48b8a3bfae6d043e331b.glb'),
                'cpu')
            data = o3d.io.read_point_cloud(os.path.join(self.base_path, 'fps_downsample/000060a495b381230860ca7315a1b585fabc651cf0833b72b6f481771cca4277_fps.ply'))
            surface = np.asarray(data.points)
            R = np.array([[1, 0, 0],
                          [0, 0, -1],
                          [0, 1, 0]])
            surface = surface @ R.T
            feat = load_file(os.path.join(self.base_path, 'features/000060a495b381230860ca7315a1b585fabc651cf0833b72b6f481771cca4277.safetensors'))['final_features']


        surface = torch.from_numpy(surface).float()
        if self.transform:
            surface = self.transform(surface)

        # faces = gt_mesh.faces
        if self.return_surface:
            return surface, vertices, faces, feat

        else:
            return vertices, faces, feat

    def __len__(self):
        if self.split != 'train':
            return len(self.models)
        else:
            return len(self.models) * self.replica
