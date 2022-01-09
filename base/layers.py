import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearWithConstraint(nn.Linear):
    def __init__(self, *config, max_norm=1, **kwconfig):
        self.max_norm = max_norm
        super(LinearWithConstraint, self).__init__(*config, **kwconfig)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(LinearWithConstraint, self).forward(x)


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *config, max_norm=1, **kwconfig):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*config, **kwconfig)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)


class ConvSamePad2d(nn.Module):
    """
    extend nn.Conv2d to support SAME padding
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, bias=True):
        super(ConvSamePad2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        # self.bias = bias
        self.conv = torch.nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            groups=self.groups,
            bias=bias)

    def forward(self, x):
        if type(self.kernel_size) != int:
            kernel_size = self.kernel_size
        else:
            kernel_size = (self.kernel_size, self.kernel_size)
        if type(self.stride) != int:
            stride = self.stride
        else:
            stride = (self.stride, self.stride)

        # net = x
        _, _, h, w = x.size()

        # Compute weight padding size
        out_dim = (w + stride[1] - 1) // stride[1]
        p = max(0, (out_dim - 1) * stride[1] + kernel_size[1] - w)
        pad_1 = p // 2
        pad_2 = p - pad_1
        w_pad_size = (pad_1, pad_2)

        # Compute height padding size
        out_dim = (h + stride[0] - 1) // stride[0]
        p = max(0, (out_dim - 1) * stride[0] + kernel_size[0] - h)
        pad_1 = p // 2
        pad_2 = p - pad_1
        h_pad_size = (pad_1, pad_2)

        # Pad
        x_pad = F.pad(x, w_pad_size + h_pad_size, "constant", 0)

        # Conv
        out = self.conv(x_pad)

        return out


class Swish(nn.Module):
    '''
    The swish layer: implements the swish activation function
    '''

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class LogVarLayer(nn.Module):
    '''
    The log variance layer: calculates the log variance of the data along given 'dim'
    (natural logarithm)
    '''

    def __init__(self, dim):
        super(LogVarLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.log(torch.clamp(x.var(dim=self.dim, keepdim=True), 1e-6, 1e6))


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


class SincConv(nn.Module):
    def __init__(
            self,
            kernel_size,
            min_hz=0,
            max_hz=100,
            min_band_width=16,
            band_width=4,
            sampling_rate=250,
            in_channels=1,
            padding='same',
            scale=1
    ):
        super().__init__()

        if in_channels != 1:
            msg = f"SincConv only support one input channel (here, in_channels = {in_channels})"
            raise ValueError(msg)
        self.out_channels = (max_hz - min_hz) // band_width
        self.kernel_size = kernel_size
        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1
        self.min_hz = min_hz * scale
        self.max_hz = max_hz * scale
        self.min_band_width = min_band_width * scale
        self.sampling_rate = sampling_rate
        self.padding = padding
        self.scale = scale

        # Initial filter-bank
        self.low_hz = nn.Parameter(torch.FloatTensor(range(min_hz, max_hz, band_width)).unsqueeze(1) * scale)
        self.band_width = nn.Parameter(torch.FloatTensor([band_width] * self.out_channels).unsqueeze(1) * scale)

        # Hamming window (computing only half of the window)
        n_lin = torch.linspace(0, (self.kernel_size / 2) - 1, steps=int((self.kernel_size / 2)))
        self.window = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_size)

        # Due to symmetry, I only need half of the time axes
        n = (self.kernel_size - 1) / 2.0
        self.n = 2 * math.pi * torch.arange(-n, 0).view(1, -1) / self.sampling_rate

    def forward(self, x):
        self.n = self.n.to(x.device)
        self.window = self.window.to(x.device)

        low = torch.clamp(self.low_hz, self.min_hz, self.max_hz - self.min_band_width) * (self.scale ** -1)
        band_width = torch.clamp(self.band_width, min=self.min_band_width) * (self.scale ** -1)
        high = torch.clamp(low + band_width, max=self.max_hz) * (self.scale ** -1)

        # Cut off frequency
        f_times_t_low = torch.matmul(low, self.n)
        f_times_t_high = torch.matmul(high, self.n)

        band_pass_left = ((torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (self.n / 2)) * self.window
        band_pass_center = 2 * band_width
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)
        band_pass = band_pass / (2 * band_width)

        self.filters = band_pass.view(self.out_channels, 1, 1, self.kernel_size)

        return F.conv2d(x, self.filters, stride=1, padding=self.padding)

    def get_band(self, digit=2):
        low = torch.clamp(self.low_hz, self.min_hz, self.max_hz - self.min_band_width) * (self.scale ** -1)
        band_width = torch.clamp(self.band_width, min=self.min_band_width) * (self.scale ** -1)
        high = low + band_width
        return [(round(float(l), digit), round(float(h), digit)) for l, h in zip(low, high)]


# class SincConv(nn.Module):
#     def __init__(
#             self,
#             kernel_size,
#             min_hz=0,
#             max_hz=100,
#             min_band_width=16,
#             sampling_rate=250,
#             in_channels=1,
#             padding='same'
#     ):
#         super().__init__()
#
#         if in_channels != 1:
#             msg = f"SincConv only support one input channel (here, in_channels = {in_channels})"
#             raise ValueError(msg)
#         self.out_channels = (max_hz - min_hz) // min_band_width
#         self.kernel_size = kernel_size
#         # Forcing the filters to be odd (i.e, perfectly symmetrics)
#         if kernel_size % 2 == 0:
#             self.kernel_size = self.kernel_size + 1
#         self.min_hz = min_hz
#         self.max_hz = max_hz
#         self.min_band_width = min_band_width
#         self.sampling_rate = sampling_rate
#         self.padding = padding
#
#         # Initial filter-bank
#         self.low_hz = nn.Parameter(torch.FloatTensor(range(min_hz, max_hz, min_band_width)).unsqueeze(1))
#         self.band_width = nn.Parameter(torch.FloatTensor([min_band_width] * self.out_channels).unsqueeze(1))
#
#         # Hamming window (computing only half of the window)
#         n_lin = torch.linspace(0, (self.kernel_size / 2) - 1, steps=int((self.kernel_size / 2)))
#         self.window = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_size)
#
#         # Due to symmetry, I only need half of the time axes
#         n = (self.kernel_size - 1) / 2.0
#         self.n = 2 * math.pi * torch.arange(-n, 0).view(1, -1) / self.sampling_rate
#
#     def forward(self, x):
#         self.n = self.n.to(x.device)
#         self.window = self.window.to(x.device)
#
#         low = torch.clamp(self.low_hz, self.min_hz, self.max_hz - self.min_band_width)
#         high = torch.clamp(self.low_hz + self.band_width, self.min_hz + self.min_band_width, self.max_hz)
#         band_width = high - low
#
#         # Cut off frequency
#         f_times_t_low = torch.matmul(low, self.n)
#         f_times_t_high = torch.matmul(high, self.n)
#
#         band_pass_left = ((torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (self.n / 2)) * self.window
#         band_pass_center = 2 * band_width
#         band_pass_right = torch.flip(band_pass_left, dims=[1])
#
#         band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)
#         band_pass = band_pass / (2 * band_width)
#
#         self.filters = band_pass.view(self.out_channels, 1, 1, self.kernel_size)
#
#         return F.conv2d(x, self.filters, stride=1, padding=self.padding)
#
#     def get_band(self, digit=2):
#         low = torch.clamp(self.low_hz, self.min_hz, self.max_hz - self.min_band_width)
#         high = torch.clamp(self.low_hz + self.band_width, self.min_hz + self.min_band_width, self.max_hz)
#         return [(round(float(l), digit), round(float(h), digit)) for l, h in zip(low, high)]


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
