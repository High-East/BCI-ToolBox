import torch
import torch.nn as nn

from base.layers import Conv2dWithConstraint, LinearWithConstraint
from utils.utils import initialize_weight

torch.set_printoptions(linewidth=1000)


class ShallowConvNet(nn.Module):
    def __init__(
            self,
            n_classes,
            input_shape,
            F1=None,
            T1=None,
            F2=None,
            P1_T=None,
            P1_S=None,
            drop_out=None,
            pool_mode=None,
            weight_init_method=None,
            last_dim=None,
    ):
        super(ShallowConvNet, self).__init__()
        b, c, s, t = input_shape
        pooling_layer = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[pool_mode]
        self.net = nn.Sequential(
            Conv2dWithConstraint(1, F1, (1, T1), max_norm=2),
            Conv2dWithConstraint(F1, F2, (s, 1), bias=False, max_norm=2),
            nn.BatchNorm2d(F2),
            ActSquare(),
            pooling_layer((1, P1_T), (1, P1_S)),
            ActLog(),
            nn.Dropout(drop_out),
            nn.Flatten(),
            LinearWithConstraint(last_dim, n_classes, max_norm=0.5)
        )

        initialize_weight(self, weight_init_method)

    def forward(self, x):
        out = self.net(x)
        return out


class ActSquare(nn.Module):
    def __init__(self):
        super(ActSquare, self).__init__()
        pass

    def forward(self, x):
        return torch.square(x)


class ActLog(nn.Module):
    def __init__(self, eps=1e-06):
        super(ActLog, self).__init__()
        self.eps = eps

    def forward(self, x):
        return torch.log(torch.clamp(x, min=self.eps))
