import os
from torch.utils.data import DataLoader
import wandb

from utils.utils import Color, read_json, print_dict
from . import datasets
from . import transforms


class DataGenerator:
    def __init__(self, config):
        if os.environ['TOTAL_RUNS'] == '1':
            print(f"\n{Color.BOLD}{Color.BLUE}Data Generator{Color.END}")
        self.config = config

        # Make DataLoader
        if config.mode == 'train':
            self.train_loader = self.make_data_loader('train')
            if config.is_val:
                self.val_loader = self.make_data_loader('val')
            self.test_loader = self.make_data_loader('test')
            # Set input_shape and n_classes
            mini_batch_shape = list(self.train_loader.dataset.x.shape)
            mini_batch_shape[0] = None
            config.input_shape = mini_batch_shape
            config.n_classes = len(self.train_loader.dataset.y.unique())
        else:
            self.test_loader = self.make_data_loader('test')

        # Print dataset information
        if os.environ['TOTAL_RUNS'] == '1':
            self.print_dataset()
            if os.environ['IS_WANDB'] == 'TRUE':
                wandb.run.summary['input_shape'] = config.input_shape
                wandb.run.summary['n_classes'] = config.n_classes

    def make_data_loader(self, phase):
        # TmpDataset
        if self.config.tmp_data:
            dataset = datasets.TmpDataset(data_shape=self.config.tmp_data)

        # FolderDataset
        elif self.config.folder:
            dataset = datasets.FolderDataset(
                subject=self.config.subject,
                folder=self.config.folder,
                metric_learning=self.config.metric_learning,
                phase=phase)

        # MOABBDataset
        else:
            if self.config.dataset == '2a':
                dataset = datasets.BNCI20142a(subject=self.config.subject, preproces_params=self.config.preprocess,
                                              phase=phase)
            elif self.config.dataset == '2b':
                dataset = datasets.BNCI20142b(subject=self.config.subject, preproces_params=self.config.preprocess,
                                              phase=phase)
            elif self.config.dataset == 'openbmi':
                dataset = datasets.OpenBMI(subject=self.config.subject, preproces_params=self.config.preprocess,
                                           phase=phase)
            elif self.config.dataset == 'cho2017':
                dataset = datasets.Cho2017(subject=self.config.subject, preproces_params=self.config.preprocess,
                                           phase=phase)
            else:
                raise ValueError(f"Not supported {self.config.dataset} yet.")

        # Apply data transform
        compose = transforms.Compose(self.config.transform)
        compose(dataset)

        return DataLoader(dataset,
                          batch_size=self.config.batch_size,
                          shuffle=True if phase == 'train' else False,
                          drop_last=False)

    def print_dataset(self):
        # TmpDataset
        if self.config.tmp_data:
            print(f"Data source: TmpData")

        # FolderDataset
        elif self.config.folder:
            print(f"Data folder: {self.config.folder}")
            try:
                data_info = read_json(os.path.join(self.config.folder, "data_info.json"))
                print_dict(data_info['preprocess'], 'Preprocess')
            except FileNotFoundError:
                pass

        # MOABBDataset
        else:
            print(f"Dataset: {self.config.dataset}")

        # Data information
        print(f"Number of classes: {self.config.n_classes}")
        if self.config.mode == 'train':
            print(f"Train set: {list(self.train_loader.dataset.x.shape)}")
            if self.config.is_val:
                print(f"Validation set: {list(self.val_loader.dataset.x.shape)}")
            print(f"Test set: {list(self.test_loader.dataset.x.shape)}")
        else:
            print(f"Test set: {list(self.test_loader.dataset.x.shape)}")
