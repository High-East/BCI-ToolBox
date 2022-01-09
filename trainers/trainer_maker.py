import os
import sys
from collections import defaultdict

import torch.nn as nn
import torch.optim as optim

from utils.utils import Color
from trainers import *

cur_modules = sys.modules[__name__].__dict__


class TrainerMaker:
    def __init__(self, config, model, data):
        if os.environ['TOTAL_RUNS'] == '1':
            print(f"\n{Color.BOLD}{Color.BLUE}Trainer Maker{Color.END}")
        if config.mode == 'train':
            criterion = self.set_criterion(config.criterion)
            optimizer = self.set_optimizer(config, model)
            scheduler = self.set_scheduler(config, optimizer)
            history = defaultdict(list)
            self.trainer = self.make_trainer(config=config,
                                             model=model,
                                             data=data,
                                             criterion=criterion,
                                             optimizer=optimizer,
                                             scheduler=scheduler,
                                             history=history)
        else:
            criterion = self.set_criterion(config.criterion)
            history = defaultdict(list)
            self.trainer = self.make_trainer(config=config,
                                             model=model,
                                             data=data,
                                             criterion=criterion,
                                             history=history)
        if os.environ['TOTAL_RUNS'] == '1':
            self.print_trainer()

    def set_criterion(self, criterion):
        criterion_list = dict(
            MSE='MSELoss',
            CEE='CrossEntropyLoss',
            Triplet='TripletMarginLoss'
        )
        get_criterion = lambda criterion: getattr(nn, criterion)()
        if type(criterion) == str:
            return get_criterion(criterion_list[criterion])
        elif type(criterion) == list:
            return [get_criterion(criterion_list[c]) for c in criterion]

    def set_optimizer(self, config, model):
        if config.opt == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=float(config.lr), weight_decay=float(config.wd))
        elif config.opt == "Adam":
            optimizer = optim.Adam(list(model.parameters()), lr=float(config.lr), weight_decay=float(config.wd))
        elif config.opt == 'AdamW':
            optimizer = optim.AdamW(list(model.parameters()), lr=float(config.lr), weight_decay=float(config.wd))
        else:
            raise ValueError(f"Not supported {config.opt}.")
        return optimizer

    def set_scheduler(self, config, optimizer):
        if config.scheduler is None:
            return None
        elif config.scheduler == 'exp':
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.gamma)
        elif config.scheduler == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
        elif config.scheduler == 'multi_step':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=config.gamma)
        elif config.scheduler == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20,
                                                             threshold=0.1, threshold_mode='abs', verbose=True)
        elif config.scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                             T_max=int(config.T_max) if (hasattr(config, 'T_max')
                                                                                         and config.T_max) else config.epochs,
                                                             eta_min=float(config.eta_min) if (
                                                                     hasattr(config, 'eta_min')
                                                                     and config.eta_min) else 0)
        else:
            raise ValueError(f"Not supported {config.scheduler}.")
        return scheduler

    def make_trainer(self, **kwargs):
        if kwargs['config'].trainer == 'cls':
            kwargs['config'].trainer = 'CLS'
        trainer = cur_modules.get(f"{kwargs['config'].trainer}Trainer")(**kwargs)
        return trainer

    def print_trainer(self):
        print(
            f"Type: {self.trainer.config.trainer}\n",
            f"Batch: size: {self.trainer.config.batch_size}\n",
            f"Epochs: {self.trainer.config.epochs}\n",
            f"Loss: {self.trainer.config.criterion}\n",
            f"Metrics: {self.trainer.config.metrics}\n",
            f"Optimizer: {self.trainer.config.opt}\n",
            f"Learning rate: {self.trainer.config.lr}\n",
            f"Weight decay: {self.trainer.config.wd}\n",
            f"Scheduler: {self.trainer.config.scheduler}\n",
            f"Eta_min: {self.trainer.config.eta_min}" if self.trainer.config.eta_min is not None else None,
            sep=""
        )
