import os
import torch
from torch.utils import data
import numpy as np
import pandas as pd
class TrellisDataset(data.Dataset):
    def __init__(self, dataset_folder, transform=None):
        self.dataset_folder = dataset_folder
        self.transform = transform

        self.csv_dir = os.path.join(self.dataset_folder, 'metadata.csv')
        self.metadata = pd.read_csv(self.csv_dir)
        self.models = self.metadata['local_path'].tolist()

        # self.models = []


    def __getitem__(self, index):

    def __len__(self):
