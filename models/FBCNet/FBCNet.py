# https://arxiv.org/pdf/2104.01233.pdf

# import sys
import torch.nn as nn
from torch.nn import functional as F

from base.layers import Conv2dWithConstraint, LinearWithConstraint, Swish, LogVarLayer
from utils.utils import initialize_weight

# current_module = sys.modules[__name__]


class FBCNet(nn.Module):
    def __init__(self,
                 n_classes,
                 input_shape,
                 m,
                 temporal_stride,
                 weight_init_method=None,
                 ):
        super().__init__()
        self.temporal_stride = temporal_stride

        batch_size, n_band, n_electrode, time_points = input_shape

        # SCB (Spatial Convolution Block)
        self.scb = nn.Sequential(
            Conv2dWithConstraint(n_band, m * n_band, (n_electrode, 1), groups=n_band, max_norm=2),
            nn.BatchNorm2d(m * n_band),
            Swish()
        )

        # Temporal Layer
        self.temporal_layer = LogVarLayer(-1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            LinearWithConstraint(n_band * m * temporal_stride, n_classes, max_norm=0.5)
        )

        initialize_weight(self, weight_init_method)

    def forward(self, x):
        out = self.scb(x)
        out = F.pad(out, (0, 3))
        out = out.reshape([*out.shape[:2], self.temporal_stride, int(out.shape[-1] / self.temporal_stride)])
        out = self.temporal_layer(out)
        out = self.classifier(out)
        return out
