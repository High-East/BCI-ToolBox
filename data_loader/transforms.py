from functools import reduce
import importlib
import numpy as np
import torch

from utils.utils import one_hot_encoding

__all__ = ('Compose', 'ToTensor', 'LabelSelection', 'ChannelSelection')


class Compose:
    def __init__(self, transforms):
        if transforms.__class__.__name__ not in ['AttrDict', 'dict']:
            raise TypeError(f"Not supported {transforms.__class__.__name__} type yet.")
        transforms_module = importlib.import_module('data_loader.transforms')
        configure_transform = lambda transform, params: getattr(transforms_module, transform)(**params) \
            if type(params).__name__ in ['dict', 'AttrDict'] else getattr(transforms_module, transform)(params)
        self.transforms = [configure_transform(transform, params) for transform, params in transforms.items()]

    def __call__(self, dataset):
        for transform in self.transforms:
            transform(dataset)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(\n'
        for transform in self.transforms:
            format_string += f'    {transform.__class__.__name__}()\n'
        format_string += ')'
        return format_string


class ToTensor:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, dataset):
        if type(dataset.x) != torch.Tensor:
            dataset.x = torch.as_tensor(dataset.x, dtype=torch.float)
        if type(dataset.y) != torch.Tensor:
            dataset.y = torch.as_tensor(dataset.y, dtype=torch.long)

    def __repr__(self):
        return self.__class__.__name__


class LabelSelection:
    def __init__(self, target_labels, is_one_hot=False):
        self.target_labels = target_labels
        self.is_one_hot = is_one_hot

    def __call__(self, dataset):
        if self.is_one_hot == 'one_hot_encoding':
            dataset.y = np.argmax(dataset.y, axis=1)

        # Select labels
        labels = ((dataset.y == label) for label in self.target_labels)
        idx = reduce(lambda x, y: x | y, labels)
        dataset.x = dataset.x[idx]
        dataset.y = dataset.y[idx]

        # Mapping labels
        for mapping, label in enumerate(np.unique(dataset.y)):
            dataset.y[dataset.y == label] = mapping

        if self.is_one_hot == 'one_hot_encoding':
            dataset.y = one_hot_encoding(dataset.y)

    def __repr__(self):
        return self.__class__.__name__


class ChannelSelection:
    def __init__(self, target_chans):
        self.target_chans = target_chans

    def __call__(self, dataset):
        target_idx = [i for i, chan in enumerate(self.target_chans) if chan in dataset.ch_names]
        dataset.x = dataset.x[..., target_idx, :]

    def __repr__(self):
        return self.__class__.__name__


class TimeSegmentation:
    def __init__(self, window_size, step_size, axis=1, merge='stack'):
        self.window_size = window_size
        self.step_size = step_size
        self.axis = axis
        self.merge = merge

    def __call__(self, dataset):
        segments = []
        times = np.arange(dataset.x.shape[-1])
        start = times[::self.step_size]
        end = start + self.window_size
        for s, e in zip(start, end):
            if e > len(times):
                break
            segments.append(dataset.x[..., s:e])
        if self.merge == 'stack':
            dataset.x = np.stack(segments, axis=self.axis)
        elif self.merge == 'concat':
            dataset.x = np.concatenate(segments, axis=self.axis)
        else:
            raise ValueError(f"{self.merge} is not supported yet.")

    def __repr__(self):
        return self.__class__.__name__
