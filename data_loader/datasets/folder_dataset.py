import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset

from data_loader import datasets
from utils.utils import read_yaml


class FolderDataset(Dataset):
    def __init__(self, folder, phase, subject, ch_names=False, metric_learning=False):
        self.metric_learning = metric_learning
        try:
            self.x = np.load(os.path.join(folder, phase, f"S{subject:02}_x.npy"))
            self.y = np.load(os.path.join(folder, phase, f"S{subject:02}_y.npy"))
        except FileNotFoundError:
            try:
                self.x = torch.load(os.path.join(folder, phase, f"S{subject:02}_x.pt"))
                self.y = torch.load(os.path.join(folder, phase, f"S{subject:02}_y.pt"))
            except FileNotFoundError:
                print(f"Folder should be included in {datasets.folder_dataset_list}.")
        if ch_names:
            self.ch_names = read_yaml(os.path.join(folder, "data_info.yaml"))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if not self.metric_learning:
            sample = [self.x[idx], self.y[idx]]
            return sample
        else:
            anchor = self.x[idx]
            anchor_label = self.y[idx]
            positive_indices = torch.where(self.y == self.y[idx])[0]
            while True:
                positive_idx = random.choice(positive_indices)
                if positive_idx != idx:
                    positive = self.x[positive_idx]
                    break
            negative = self.x[random.choice(torch.where(self.y != self.y[idx])[0])]
            sample = (anchor, anchor_label, positive, negative)
            return sample
