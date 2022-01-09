from .tmp_dataset import TmpDataset
from .folder_dataset import FolderDataset
from .bnci2014 import BNCI20142a, BNCI20142b
from .openbmi import OpenBMI
from .cho2017 import Cho2017

# http://moabb.neurotechx.com/docs/datasets.html

__all__ = (
    'folder_dataset_list',
    'TmpDataset',
    'FolderDataset',
    'BNCI20142a',
    'BNCI20142b',
    'OpenBMI',
    'Cho2017')

folder_dataset_list = (
    'data_loader/datasets/2a/4-classes/val-10',
    'data_loader/datasets/2a/4-classes/dahyun/20-shot/session1',
    'data_loader/datasets/2a/4-classes/dahyun/20-shot/session2',
    'data_loader/datasets/2a/4-classes/dahyun/36-shot/session1',
    'data_loader/datasets/2a/4-classes/dahyun/36-shot/session2',
)
