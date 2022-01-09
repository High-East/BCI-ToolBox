import torch.nn as nn

from base.layers import Conv2dWithConstraint, LinearWithConstraint
from utils.utils import initialize_weight


class EEGNet(nn.Module):
    def __init__(
            self,
            n_classes,
            input_shape,
            F1=None,
            D=None,
            F2=None,
            T1=None,
            T2=None,
            P1=None,
            P2=None,
            drop_out=None,
            pool_mode=None,
            weight_init_method=None,
    ):
        super().__init__()
        _, c, s, t = input_shape
        pooling_layer = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[pool_mode]

        if F2 == 'auto':
            F2 = F1 * D

        # Spectral
        self.spectral = nn.Sequential(
            nn.Conv2d(1, F1, (1, T1), bias=False, padding='same'),
            nn.BatchNorm2d(F1))

        # Spatial
        self.spatial = nn.Sequential(
            Conv2dWithConstraint(F1, F1 * D, (s, 1), bias=False, groups=F1),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            pooling_layer((1, P1)),
            nn.Dropout(drop_out)
        )

        # Temporal
        self.temporal = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, T2), bias=False, padding='same', groups=F1 * D),
            nn.Conv2d(F2, F2, 1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            pooling_layer((1, P2)),
            nn.Dropout(drop_out)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            LinearWithConstraint(F2 * 8, n_classes, max_norm=0.25)
        )

        initialize_weight(self, weight_init_method)

    def forward(self, x):
        out = self.spectral(x)
        out = self.spatial(out)
        out = self.temporal(out)
        out = self.classifier(out)
        return out
