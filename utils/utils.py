import os
from functools import wraps
import sys
import string
import argparse
import json
import pickle
import yaml
import re
import time
import random
import itertools
import importlib
import numpy as np
import scipy.signal as sig
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn


def convert_time(sec):
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    print(f"Total time: {h:02}:{m:02}:{s:02}")


def set_random_seed(seed, device, verbose=False):
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
    if verbose:
        print(f"[✓] set random seed: {seed}")


def str2list(string):
    if string == 'all':
        return 'all'
    else:
        return string.split(",")


def str2list_int(string):
    if string == 'all':
        return 'all'
    else:
        return list(map(int, string.split(",")))


def make_dir(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Creating directory. " + directory)
    return directory


def print_dict(dictionary, name=None):
    if name:
        print(f"{name}(")
    for key, value in dictionary.items():
        print(f"    {key}={value}")
    print(")")


def print_update(sentence, i=None):
    """
    Args:
        sentence: sentence
        i: index in for loop

    Returns:
    """

    # print(sentence, end='') if i == 0 else print('\r' + sentence, end='')
    print('\r' + sentence, end='')


def progress_bar(idx, length=20, symbol='>'):
    if idx < 1 or not isinstance(idx, int):
        raise ValueError("idx should be at least 1 integers.")
    if idx < length:
        bar = '[' + '=' * idx + symbol + ' ' * (length - idx) + ']'
    else:
        bar = '[' + '=' * (length + 1) + ']'
    return bar


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def write_json(path, data, cls=MyEncoder):
    with open(path, "w") as json_file:
        json.dump(data, json_file, cls=cls)


def read_json(path):
    with open(path, "r") as file:
        data = json.load(file)
    return data


def write_pickle(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def read_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def read_yaml(path):
    with open(path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data


def write_yaml(path, data):
    with open(path, 'w') as f:
        yaml.dump(data, f)


class AttrDict(dict):
    def __init__(self, *config, **kwconfig):
        super(AttrDict, self).__init__(*config, **kwconfig)
        self.__dict__ = self
        for key in self:
            if type(self[key]) == dict:
                self[key] = AttrDict(self[key])

    def __getattr__(self, item):
        return None

    def get_values(self, keys):
        return {key: self.get(key) for key in keys}

    def dict(self):
        dictionary = dict(self)
        for key in dictionary:
            if type(dictionary[key]).__name__ == 'AttrDict':
                dictionary[key] = dict(dictionary[key])
        return dictionary


def print_off():
    sys.stdout = open(os.devnull, 'w')


def print_on():
    sys.stdout = sys.__stdout__


def one_hot_encoding(arr):
    num = len(np.unique(arr))
    encoding = np.eye(num)[arr]
    return encoding


def order_change(arr, order):
    arr = list(arr)
    tmp = arr[order[0]]
    arr[order[0]] = arr[order[1]]
    arr[order[1]] = tmp
    return arr


def transpose_np(tensor, order):
    return np.transpose(tensor, order_change(np.arange(len(tensor.shape)), order))


def import_model(config):
    module = importlib.import_module(f'models.{config.model}_model')
    model = getattr(module, config.model)(input_shape=config.input_shape,
                                          n_classes=config.n_classes,
                                          **config.arch)
    return model


def initialize_weight(model, method):
    method = dict(normal=['normal_', dict(mean=0, std=0.01)],
                  xavier_uni=['xavier_uniform_', dict()],
                  xavier_normal=['xavier_normal_', dict()],
                  he_uni=['kaiming_uniform_', dict()],
                  he_normal=['kaiming_normal_', dict()]).get(method)
    if method is None:
        return None

    for module in model.modules():
        # LSTM
        if module.__class__.__name__ in ['LSTM']:
            for param in module._all_weights[0]:
                if param.startswith('weight'):
                    getattr(nn.init, method[0])(getattr(module, param), **method[1])
                elif param.startswith('bias'):
                    nn.init.constant_(getattr(module, param), 0)
        else:
            if hasattr(module, "weight"):
                # Not BN
                if not ("BatchNorm" in module.__class__.__name__):
                    getattr(nn.init, method[0])(module.weight, **method[1])
                # BN
                else:
                    nn.init.constant_(module.weight, 1)
                if hasattr(module, "bias"):
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)


def get_config(save_path):
    config = read_json(os.path.join(save_path, "config.json"))
    return config


def pretrained_model(path):
    try:
        model = read_pickle(os.path.join(path, 'model.pk'))
    except FileNotFoundError:
        raise FileNotFoundError
    save_path = set_pretrained_path(path)
    model = load_model(model, save_path)
    return model


def load_model(model, path, load_range='all'):
    checkpoint = torch.load(path, map_location='cpu')
    if next(iter(checkpoint['model_state_dict'].keys())).startswith('module'):
        new_state_dict = dict()
        for k, v in checkpoint['model_state_dict'].items():
            new_key = k[7:]
            new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict, strict=True)
    else:
        if load_range == 'all':
            target_module = set(map(lambda x: x.split(".")[0], checkpoint['model_state_dict'].keys()))
            print(f"Target module: {target_module}")
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        else:
            target_params = {k: v for k, v in checkpoint['model_state_dict'].items() if
                             k.split(".")[0] in load_range}
            model.load_state_dict(target_params, strict=False)
    return model


def set_pretrained_path(path):
    if not path.endswith('.tar'):
        tar = listdir_sort(os.path.join(path, 'checkpoints'))[-1]
        path = os.path.join(path, 'checkpoints', tar)
    return path


def listdir_sort(path):
    return sort(os.listdir(path))


def sort(array):
    '''
    sort exactly for list or array which element is string
    example: [1, 10, 2, 4] -> [1, 2, 4, 10]
    '''
    str2int = lambda string: int(string) if string.isdigit() else string
    key = lambda key: [str2int(x) for x in re.split("([0-9]+)", key)]
    return sorted(array, key=key)


def timeit(func):
    start = time.time()

    def decorator(*config):
        _return = func(*config)
        convert_time(time.time() - start)
        return _return

    return decorator


def guarantee_numpy(data):
    data_type = type(data)
    if data_type == torch.Tensor:
        device = data.device.type
        if device == 'cpu':
            data = data.detach().numpy()
        else:
            data = data.detach().cpu().numpy()
        return data
    elif data_type == np.ndarray or data_type == list:
        return data
    else:
        raise ValueError("Check your data type.")


def band_list(string):
    if string == 'all':
        return [[0, 4], [4, 7], [7, 13], [13, 30], [30, 42]]
    lst = string.split(",")
    assert len(lst) % 2 == 0, "Length of the list must be even number."
    it = iter(lst)
    return [list(map(int, itertools.islice(it, i))) for i in ([2] * (len(lst) // 2))]


class Color:
    # print(Color.BOLD + 'Hello World !' + Color.END)
    RED = '\033[91m'
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    PURPLE = '\033[95m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

    @staticmethod
    def show():
        print(
            'Color(',
            f'    {Color.RED}RED{Color.END}',
            f'    {Color.YELLOW}YELLOW{Color.END}',
            f'    {Color.GREEN}GREEN{Color.END}',
            f'    {Color.BLUE}BLUE{Color.END}',
            f'    {Color.CYAN}CYAN{Color.END}',
            f'    {Color.DARKCYAN}DARKCYAN{Color.END}',
            f'    {Color.PURPLE}PURPLE{Color.END}',
            f'    {Color.BOLD}BOLD{Color.END}',
            f'    {Color.UNDERLINE}UNDERLINE{Color.END}',
            f'    {Color.END}END{Color.END}',
            ')',
            sep='\n'
        )


def get_plv(sig1: np.ndarray, sig2: np.ndarray):
    sig1_hill = sig.hilbert(sig1)
    sig2_hill = sig.hilbert(sig2)
    phase_1 = np.angle(sig1_hill)
    phase_2 = np.angle(sig2_hill)
    phase_diff = phase_1 - phase_2
    plv = np.abs(np.mean([np.exp(complex(0, phase)) for phase in phase_diff]))
    return plv


def plv_n_dim(array: np.ndarray):
    """

    Parameters
    ----------
    array: [..., channels, times]

    Returns
    -------

    """
    tensor = np.angle(sig.hilbert(array))
    tensor = np.exp(tensor * 1j)
    plv = np.abs((tensor @ (np.transpose(tensor, order_change(np.arange(len(tensor.shape)), [-1, -2])) ** -1))
                 / np.size(tensor, -1))
    return plv


def corr_n_dim(array: np.ndarray):
    """

    Parameters
    ----------
    array: [..., channels, times]

    Returns: channels * channels correlation coefficient matrix
    -------

    """
    mean = array.mean(axis=-1, keepdims=True)
    tensor2 = array - mean
    tensor3 = tensor2 @ np.transpose(tensor2, order_change(np.arange(len(tensor2.shape)), [-1, -2]))
    tensor4 = np.sqrt(np.expand_dims(np.diagonal(tensor3, axis1=-2, axis2=-1), axis=-1) @ np.expand_dims(
        np.diagonal(tensor3, axis1=-2, axis2=-1), axis=-2))
    corr = tensor3 / tensor4
    corr = np.nan_to_num(corr, nan=0.0)
    return corr


def get_random_string(length):
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for _ in range(length))
    return result_str


def get_current_time():
    now = time.localtime()
    current_time = f"{now.tm_year}{now.tm_mon}{now.tm_mday}_{now.tm_hour:02d}{now.tm_min:02d}{now.tm_sec:02d}"
    return current_time


# def save_config(func):
#     def parsing():
#         parser = argparse.ArgumentParser()
#         parser.add_argument('--config_file')
#
#         return parser.parse_known_args()[0]
#
#     @wraps(func)
#     def inner():
#         args = parsing()
#         read_yaml(args.config_file)
#
#         # Login wandb
#         if args.use_wandb or args.sweep_file:
#             wandb_login()
#             # Start sweeps
#             if args.sweep_file:
#                 os.environ['USE_SWEEP'] = 'True'
#                 start_sweep(args.sweep_file, func)
#             else:
#                 os.environ['USE_SWEEP'] = 'False'
#                 func()
#             # Write summary
#             # write_summary()  # 여기가 아니라 다른 곳에 있어야 하넹;; 여기 있으면 sweep에 적용이 안됨!
#         else:
#             func()
#
#     return inner

def figlet(func):
    @wraps(func)
    def inner():
        os.system("cat /home/user/disk2/ko/donghee.txt")
        func()

    return inner



