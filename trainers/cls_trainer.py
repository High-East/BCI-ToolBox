import os
import math
from collections import defaultdict

import torch
import wandb

from utils.calculator import Calculator
from utils.utils import (
    Color,
    make_dir,
    progress_bar,
    print_update,
    write_json
)


class CLSTrainer:
    """
    Classification Trainer
    """

    def __init__(self, config, model, data, criterion, optimizer=None, scheduler=None, history=None):
        self.config = config
        self.model = model
        self.data = data
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.history = history
        self.calculator = Calculator()

    def train(self):
        print(f"\n{Color.BOLD}{Color.BLUE}Start Run{Color.END}")
        print(f"Subject: {self.config.subject}")
        print(f"Save dir: {self.config.save_dir}")

        # W&B: watch model
        self.watch_model()

        # Train
        for epoch in range(1, self.config.epochs + 1):
            self.step_epoch(train_phase='train', epoch=epoch)

        # Test
        if self.config.is_test:
            self.step_epoch(train_phase='test')

        # Save history
        self.save_history()

    def step_epoch(self, train_phase='train', mode='train', epoch=None):
        history_mini_batch = defaultdict(list)

        # Test phase
        if mode == 'test' or train_phase == 'test':
            self.step_mini_batch('test', history_mini_batch)

        # Train phase
        else:
            self.step_mini_batch('train', history_mini_batch)
            if self.config.is_val:
                self.step_mini_batch('val', history_mini_batch)

        # Write history per epoch
        self.write_history(history_mini_batch)

        # Print history
        self.print_progress_bar(epoch=epoch, train_phase=train_phase, mode=mode)

        # Log wandb
        self.log_wandb(epoch, train_phase)

        # Save checkpoint & update scheduler
        if mode == 'train' and train_phase == 'train':
            self.save_checkpoint(epoch=len(self.history['train_loss']))
            self.update_scheduler()

    def step_mini_batch(self, data_type, history_mini_batch):
        if data_type == 'train':
            self.model.train()
            torch.set_grad_enabled(True)
        else:
            self.model.eval()
            torch.set_grad_enabled(False)

        for i, data in enumerate(getattr(self.data, f"{data_type}_loader")):
            # Forward
            inputs, labels = data[0].to(self.model.device), data[1].to(self.model.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # Backward
            if data_type == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Calculate log per mini-batch
            log = self.calculator.calculate(self.config.metrics, loss, labels, outputs, acc_count=True)

            # Record history per mini-batch
            self.record_history(log, history_mini_batch, data_type=data_type)

    def record_history(self, log, history, data_type):
        for metric in log:
            history[f'{data_type}_{metric}'].append(log[metric])

    def write_history(self, history):
        for metric in history:
            if metric.endswith('acc'):
                n_samples = len(getattr(self.data, f"{metric.split('_')[0]}_loader").dataset.y)
                self.history[metric].append((sum(history[metric]) / n_samples))
            else:
                self.history[metric].append(sum(history[metric]) / len(history[metric]))

    def print_progress_bar(self, epoch=None, train_phase='train', mode='train'):
        if mode == 'train':
            if train_phase == 'test':
                epoch = self.config.epochs
            sentence = f"[{str(epoch).rjust(len(str(self.config.epochs)))}/{self.config.epochs}] "
            sentence += f"{progress_bar(math.ceil(epoch * 20 / self.config.epochs))} "
            for metric in self.history:
                sentence += f"{metric}={self.history[metric][-1]:0.4f} "
            if train_phase == 'train':
                sentence += f"lr={self.optimizer.state_dict()['param_groups'][0]['lr']:0.8f} "
                sentence += f"wd={self.optimizer.state_dict()['param_groups'][0]['weight_decay']}"
            print_update(sentence)
        else:
            sentence = ""
            for metric in self.history:
                sentence += f"{metric}={self.history[metric][-1]:0.4f} "
            print(sentence)

    def log_wandb(self, epoch, phase):
        if os.environ['IS_WANDB'] == 'TRUE':
            if phase == 'train':
                log = {
                    'epoch': epoch,
                    'lr': self.optimizer.state_dict()['param_groups'][0]['lr']
                }
                log.update({f"S{self.config.subject:02}_{k}": v[-1] for k, v in self.history.items()})
                wandb.run.log(log)
            else:
                wandb.run.summary[f"S{self.config.subject:02}"] = self.history['test_acc'][0]
                if os.environ['TOTAL_RUNS'] == '1':
                    wandb.run.summary['acc_sum'] = str(self.history['test_acc'][0])
                else:
                    wandb.run.summary['acc_sum'] = str(
                        float(wandb.run.summary['acc_sum']) + self.history['test_acc'][0])
                if int(os.environ['TOTAL_RUNS']) == len(self.config.target_subject):
                    wandb.run.summary['Mean'] = float(wandb.run.summary['acc_sum']) / len(self.config.target_subject)

    def save_checkpoint(self, epoch):
        make_dir(os.path.join(self.config.save_dir, "checkpoints"))
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None
        }, os.path.join(self.config.save_dir, f"checkpoints/{epoch}.tar"))
        if epoch >= 6:
            os.remove(os.path.join(self.config.save_dir, f"checkpoints/{epoch - 5}.tar"))

    def update_scheduler(self):
        if self.scheduler:
            self.scheduler.step()

    def save_history(self):
        write_json(os.path.join(self.config.save_dir, "history.json"), self.history)

    def test(self):
        print(f"\n{Color.BOLD}{Color.BLUE}Start Run{Color.END}")
        print(f"Subject: {self.config.subject}")
        print(f"Pretrained path: {self.config.pretrained_path}")

        self.step_epoch(mode='test')

    def watch_model(self):
        """
        Not supported for watch model.
        """
        # if os.environ['USE_WANDB'] == 'TRUE':
        #     wandb.run.watch(
        #         self.model,
        #         log='all',  # "log all" save weights and gradients.
        #         log_freq=self.config.log_freq,
        #         idx=self.config.subject
        #     )
        pass

    def analysis(self):
        """
        Not supported anaylsis mode.
        """
        # from utils.painter import Painter
        # if self.config.analysis_method == 't_sne':
        #     self.model.eval()
        #     x = self.data.test_loader.dataset.x.to(self.model.device)
        #     for name, module in self.model.named_children():
        #         x = module(x)
        #         painter = Painter(x=x, y=self.data.test_loader.dataset.y)
        #         painter.t_sne(n_components=2, seed=42, title=f"S{self.config.subject:02}-{name}")
        pass
