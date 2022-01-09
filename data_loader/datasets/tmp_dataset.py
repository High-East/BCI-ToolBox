import torch
from torch.utils.data import Dataset


class TmpDataset(Dataset):
    def __init__(self, data_shape):
        self.x = torch.rand(data_shape)
        self.y = torch.randint(high=2, size=data_shape[0:1])  # 0 or 1

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = [self.x[idx], self.y[idx]]
        return sample
