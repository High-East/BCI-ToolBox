import os
import sys
import shutil
import argparse
from functools import (
    wraps,
    partial
)
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import mne
import braindecode
import wandb

from utils.utils import (
    make_dir,
    read_yaml,
    AttrDict,
    Color,
    write_json,
    read_json
)


def load_metadata(run):
    @wraps(run)
    def inner():
        print(f"\n{Color.BOLD}{Color.BLUE}Construct pipeline{Color.END}")

        # Initialize environment variables
        init_environment()

        # Make base config
        base_config = make_base_config()

        # Run
        run(base_config)

    return inner


def check_wandb(run):
    @wraps(run)
    def inner(base_config):
        print(f"\n{Color.BOLD}{Color.BLUE}Check W&B{Color.END}")

        # Login wandb
        login_wandb()

        # Run sweep agent
        if base_config.sweep_file:
            print("[✓] Create sweep agent")
            os.environ['IS_SWEEP'] = 'TRUE'
            if base_config.sweep_type:
                os.environ['SWEEP_TYPE'] = base_config.sweep_type
            sweep_file = read_yaml(base_config.sweep_file)
            sweep_id = wandb.sweep(sweep=sweep_file['sweep_config'], project=base_config.project)
            print(f"\n{Color.BOLD}{Color.GREEN}Run Sweep{Color.END}")
            wandb.agent(sweep_id, partial(run, base_config), count=sweep_file['count'])
        else:
            run(base_config)

    return inner


def prepare_run(run):
    @wraps(run)
    def inner(base_config):
        print(f"\n{Color.BOLD}{Color.BLUE}Prepare run{Color.END}")
        os.environ['TOTAL_RUNS'] = '0'

        # Mode: train
        if os.environ['IS_TEST'] == 'FALSE':

            # Make base save directory
            base_config.base_save_dir = make_base_save_dir(base_config.save_dir)

            # Save metadata: config file, arch file, sweep file
            save_metadata(base_config)

            # Set random seed
            if base_config.seed:
                set_random_seed(base_config.seed, base_config.device)

            # Set target subjects
            if base_config.total_subject:
                base_config.target_subject = list(range(1, int(base_config.total_subject) + 1))

            # Init wandb
            config = init_wandb(base_config)

            # Set run subject
            for config.subject in config.target_subject:
                os.environ['TOTAL_RUNS'] = str(int(os.environ['TOTAL_RUNS']) + 1)

                # Make save directory
                config.save_dir = make_dir(os.path.join(config.base_save_dir, str(config.subject)))

                # Save config file
                write_json(os.path.join(config.save_dir, 'config.json'), config)

                # Run
                if os.environ['TOTAL_RUNS'] == '1':
                    print_system(config.device)
                run(config)
                print("")

            if os.environ['IS_SWEEP'] == 'TRUE':
                wandb.finish()
                print(f"\n{Color.BOLD}{Color.GREEN}Run Sweep{Color.END}")

        # Mode: test
        else:
            # Set target subjects
            if base_config.all_subject:
                base_config.target_subject = sorted(
                    [int(directory) for directory in os.listdir(base_config.load_path) if directory.isdigit()]
                )

            # Set run subject
            for base_config.subject in base_config.target_subject:
                os.environ['TOTAL_RUNS'] = str(int(os.environ['TOTAL_RUNS']) + 1)

                # Load config file
                config = AttrDict(
                    read_json(os.path.join(base_config.load_path, str(base_config.subject), 'config.json'))
                )
                config.mode = 'test'
                config.pretrained_path = os.path.join(base_config.load_path, str(base_config.subject))

                # Run
                if os.environ['TOTAL_RUNS'] == '1':
                    print_system(config.device)
                run(config)
                print("")

    return inner


def init_environment():
    os.environ['IS_TEST'] = 'FALSE'
    os.environ['IS_WANDB'] = 'TRUE'
    os.environ['IS_SWEEP'] = 'FALSE'
    os.environ['SWEEP_TYPE'] = 'NULL'
    os.environ['WANDB_API_KEY'] = 'NULL'
    print("[✓] init environment")


def make_base_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', required=True)
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--device', default=0, type=int, help="cpu or gpu number")

    # Parsing
    config = vars(parser.parse_args())
    config = AttrDict(config)

    # Read config file
    config.update(read_yaml(config['config_file']))

    # Read arch file
    if config.arch_file:
        config.arch.update(read_yaml(config['arch_file']))

    # Check mode
    if config.mode == 'test':
        os.environ['IS_TEST'] = 'TRUE'

    # Check W&B
    if config.no_wandb or os.environ['IS_TEST'] == 'TRUE':
        os.environ['IS_WANDB'] = 'FALSE'

    print("[✓] make base_config")
    return config


def login_wandb():
    if os.environ['IS_WANDB'] == 'TRUE':
        print("[✓] login wandb")
        try:
            os.environ['WANDB_API_KEY'] = read_yaml('./configs/wandb_key.yaml')['key']
            wandb.login()
        except FileNotFoundError:
            raise ValueError("Make './configs/wandb_key.yaml' file including wandb key.")
    else:
        print("[x] login wandb")


def make_base_save_dir(save_dir):
    try:
        sub_dir = len(os.listdir(f"./result/{save_dir}"))
    except FileNotFoundError:
        make_dir(f"./result/{save_dir}")
        sub_dir = len(os.listdir(f"./result/{save_dir}"))
    base_save_dir = make_dir(f"./result/{save_dir}/{sub_dir}")
    print(f"[✓] make base save directory: {Color.YELLOW}{base_save_dir}{Color.END}")
    return base_save_dir


def save_metadata(config):
    """
    Save three files (config_file, arch_file, sweep_file)
    """
    msg = []
    metadata_dir = make_dir(os.path.join(config.base_save_dir, "metadata"))
    if config.config_file:
        shutil.copy(config.config_file, os.path.join(metadata_dir, 'config_file.yaml'))
        msg.append('config file')
    if config.arch_file:
        shutil.copy(config.arch_file, os.path.join(metadata_dir, 'arch_file.yaml'))
        msg.append('arch file')
    if config.sweep_file:
        shutil.copy(config.sweep_file, os.path.join(metadata_dir, 'sweep_file.yaml'))
        msg.append('sweep file')
    print(f"[✓] save metadata: {', '.join(msg)}")


def set_random_seed(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Multi-GPU
    if device == "multi":
        torch.cuda.manual_seed_all(seed)
    # Single-GPU
    else:
        torch.cuda.manual_seed(seed)
    cudnn.benchmark = False  # If you want to set randomness, cudnn.benchmark = False
    cudnn.deterministic = True  # If you want to set randomness, cudnn.benchmark = True
    print(f"[✓] set random seed: {Color.YELLOW}{seed}{Color.END}")


def init_wandb(base_config):
    if os.environ['IS_WANDB'] == 'TRUE':
        print("[✓] init wandb")
        wandb.init(
            project=str(base_config.project) if os.environ['IS_SWEEP'] == 'FALSE' else None,
            name=os.path.basename(base_config.base_save_dir),
            dir=base_config.base_save_dir,
            notes=base_config.notes,
            id=base_config.id,
            resume=base_config.resume,
            tags=base_config.tags
        )
        if os.environ['IS_SWEEP'] == 'TRUE' and os.environ['SWEEP_TYPE'] == 'arch':
            base_config.arch.update(wandb.run.config)

        # Update config
        wandb.run.config.update(base_config, allow_val_change=True)
        return AttrDict(wandb.run.config)
    else:
        return base_config


def print_system(device):
    print(
        f"\n{Color.BOLD}{Color.BLUE}System environments{Color.END}",
        f"PID: {os.getpid()}",
        f"Device: {torch.cuda.get_device_name()} - {device}" if device != 'cpu'
        else 'Device: CPU',
        sep='\n'
    )
    print(
        f"\n{Color.BOLD}{Color.BLUE}Requirements{Color.END}",
        f"Python: {sys.version.split(' ')[0]}",
        f"PyTorch version: {torch.__version__}",
        f"Brainecode version: {braindecode.__version__}",
        f"MNE version: {mne.__version__}",
        sep='\n'
    )
