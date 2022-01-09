import os
import sys
import torch
from torchinfo import summary
import wandb

from utils.utils import pretrained_model, write_pickle, Color
from models import *

cur_modules = sys.modules[__name__].__dict__


class ModelBuilder:
    def __init__(self, config):
        if os.environ['TOTAL_RUNS'] == '1':
            print(f"\n{Color.BOLD}{Color.BLUE}Model Builder{Color.END}")
            print(f"Type: {config.model}")

        self.model = self.build_model(config)
        self.set_device(config.device)
        self.model_summary(config, self.model)

    def build_model(self, config):
        if config.mode == 'train':
            model = cur_modules.get(config.model)(
                input_shape=config.input_shape,
                n_classes=config.n_classes,
                **config.arch
            )
            write_pickle(os.path.join(config.save_dir, "model.pk"), model)
        else:
            model = pretrained_model(config.pretrained_path)
        return model

    def set_device(self, device_ids):
        if device_ids == 'cpu':
            device = torch.device("cpu")
        else:
            if not torch.cuda.is_available():
                raise ValueError("Check GPU")
            # Single GPU
            if isinstance(device_ids, int):
                device = torch.device(f'cuda:{device_ids}')
                torch.cuda.set_device(device)  # If you want to check device, use torch.cuda.current_device().
                self.model.cuda()
            # Multi GPU
            else:
                device = torch.device(f'cuda:{device_ids[0]}')
                torch.cuda.set_device(device)
                self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
                self.model.cuda()
        # Print device
        self.model.device = device

    def model_summary(self, config, model):
        results = summary(model, col_names=["kernel_size", "num_params"],
                          device=model.device if not model.device == 'multi' else torch.device("cuda:0"),
                          verbose=1 if config.summary else 0)
        config.model_params = results.trainable_params
        if os.environ['TOTAL_RUNS'] == '1':
            print(f"Model parameters: {config.model_params}")
            if os.environ['IS_WANDB'] == 'TRUE':
                wandb.run.summary['model_params'] = config.model_params
