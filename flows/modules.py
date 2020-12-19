import torch
import torch.nn as nn
from torch.nn.utils import weight_norm as WeightNorm

from .spectral_norm import SpectralNorm


def deriv_sigmoid(x):
    """ derivative of sigmoid """
    sigma = torch.sigmoid(x)
    return sigma * (1.0 - sigma)


def deriv_logit(x, eps=1.0e-8):
    """ derivative of logit """
    y = torch.logit(torch.clamp(x, eps, 1.0 - eps))
    return 1.0 / deriv_sigmoid(y)


def deriv_tanh(x):
    """ derivative of tanh """
    y = torch.tanh(x)
    return 1.0 - y * y


def deriv_arctanh(x, eps=1.0e-8):
    """ derivative of arctanh """
    x = torch.clamp(x, -1.0 + eps, 1.0 - eps)
    return 1.0 / (1.0 - x * x)


def mix_log_cdf(x, pi, mu, s):
    x = x.unsqueeze(1)
    x = pi * torch.sigmoid((x - mu) * torch.exp(-s))
    x = torch.sum(x, dim=1)
    return x


def deriv_mix_log_cdf(x, pi, mu, s):
    x = x.unsqueeze(1)
    w = torch.exp(-s)
    x = pi * deriv_sigmoid((x - mu) * w) * w
    x = torch.sum(x, dim=1)
    return x


class Sigmoid(nn.Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x, log_det_jacobians):
        log_det = torch.log(deriv_sigmoid(x))
        log_det = torch.sum(log_det.view(x.size(0), -1), dim=1)
        return torch.sigmoid(x), log_det_jacobians + log_det

    def backward(self, x, log_det_jacobians):
        x = torch.clamp(x, 1.0e-8, 1.0 - 1.0e-8)
        log_det = torch.log(deriv_logit(x))
        log_det = torch.sum(log_det.view(x.size(0), -1), dim=1)
        return torch.logit(x), log_det_jacobians + log_det


class Logit(nn.Module):
    def __init__(self):
        super(Logit, self).__init__()

    def forward(self, x, log_det_jacobians):
        x = torch.clamp(x, 1.0e-8, 1.0 - 1.0e-8)
        log_det = torch.log(deriv_logit(x))
        log_det = torch.sum(log_det.view(x.size(0), -1), dim=1)
        return torch.logit(x), log_det_jacobians + log_det

    def backward(self, x, log_det_jacobians):
        log_det = torch.log(deriv_sigmoid(x))
        log_det = torch.sum(log_det.view(x.size(0), -1), dim=1)
        return torch.sigmoid(x), log_det_jacobians + log_det


class Tanh(nn.Module):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x, log_det_jacobians):
        log_det = torch.log(deriv_tanh(x))
        log_det = torch.sum(log_det.view(x.size(0), -1), dim=1)
        return torch.tanh(x), log_det_jacobians + log_det

    def backward(self, x, log_det_jacobians):
        log_det = torch.log(deriv_arctanh(x))
        log_det = torch.sum(log_det.view(x.size(0), -1), dim=1)
        return torch.arctanh(x), log_det_jacobians + log_det


class Arctanh(nn.Module):
    def __init__(self):
        super(Arctanh, self).__init__()

    def forward(self, x, log_det_jacobians):
        log_det = torch.sum(torch.log(deriv_arctanh(x)), dim=1)
        return torch.arctanh(x), log_det_jacobians + log_det

    def backward(self, x, log_det_jacobians):
        log_det = torch.sum(torch.log(deriv_tanh(x)), dim=1)
        return torch.tanh(x), log_det_jacobians + log_det


class MixLogCDF(nn.Module):
    def __init__(self):
        super(MixLogCDF, self).__init__()

    def forward(self, x, pi, mu, s, log_det_jacobians):
        log_det = torch.log(deriv_mix_log_cdf(x, pi, mu, s))
        log_det = torch.sum(log_det.view(x.size(0), -1), dim=1)
        return mix_log_cdf(x, pi, mu, s), log_det_jacobians + log_det

    def backward(self, x, pi, mu, s, log_det_jacobians):
        lo = torch.full_like(x, -1.0e3)
        hi = torch.full_like(x, 1.0e3)
        for it in range(100):
            mid = (lo + hi) * 0.5
            val = mix_log_cdf(mid, pi, mu, s)
            lo = torch.where(val < x, mid, lo)
            hi = torch.where(val > x, mid, hi)

            if torch.all(torch.abs(hi - lo) < 1.0e-4):
                break

        x = (lo + hi) * 0.5
        log_det = torch.log(deriv_mix_log_cdf(x, pi, mu, s))
        log_det = torch.sum(log_det.view(x.size(0), -1), dim=1)

        return x, log_det_jacobians - log_det


class ResBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock1d, self).__init__()

        self.net = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            WeightNorm(nn.Linear(in_channels, out_channels)),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            WeightNorm(nn.Linear(out_channels, out_channels)),
        )

        if in_channels != out_channels:
            self.bridge = WeightNorm(nn.Linear(in_channels, out_channels))
        else:
            self.bridge = nn.Sequential()

    def forward(self, x):
        y = self.net(x)
        x = self.bridge(x)
        return x + y


class ResBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock2d, self).__init__()

        self.net = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            WeightNorm(nn.Conv2d(in_channels, out_channels, 3, 1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            WeightNorm(nn.Conv2d(out_channels, out_channels, 3, 1, 1)),
        )

        if in_channels != out_channels:
            self.bridge = WeightNorm(nn.Conv2d(in_channels, out_channels, 3, 1, 1))
        else:
            self.bridge = nn.Sequential()

    def forward(self, x):
        y = self.net(x)
        x = self.bridge(x)
        return x + y


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, base_filters=32):
        super(MLP, self).__init__()

        self.in_block = nn.Sequential(
            WeightNorm(nn.Linear(in_channels, base_filters)),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True),
            WeightNorm(nn.Linear(base_filters, base_filters)),
        )

        self.mid_block = nn.Sequential(
            ResBlock1d(base_filters, base_filters),
            ResBlock1d(base_filters, base_filters),
        )

        self.out_block = nn.Sequential(
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True),
            WeightNorm(nn.Linear(base_filters, out_channels)),
        )

    def forward(self, x):
        x = self.in_block(x)
        x = self.mid_block(x)
        return self.out_block(x)


class ConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_filters=32):
        super(ConvNet, self).__init__()

        self.in_block = nn.Sequential(
            WeightNorm(nn.Conv2d(in_channels, base_filters, 3, 1, 1)),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            WeightNorm(nn.Conv2d(base_filters, base_filters, 3, 1, 1)),
        )

        self.mid_block = nn.Sequential(
            ResBlock2d(base_filters, base_filters),
            ResBlock2d(base_filters, base_filters),
        )

        self.out_block = nn.Sequential(
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            WeightNorm(nn.Conv2d(base_filters, out_channels, 3, 1, 1)),
        )

    def forward(self, x):
        x = self.in_block(x)
        x = self.mid_block(x)
        return self.out_block(x)


class Actnorm(nn.Module):
    def __init__(self, n_channels):
        super(Actnorm, self).__init__()
        self.n_channels = n_channels
        self.log_scale = nn.Parameter(torch.zeros(n_channels), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(n_channels), requires_grad=True)
        self.initialized = False

    def forward(self, z, log_det_jacobians):
        if not self.initialized:
            self.log_scale.data.copy_(-torch.log(z.std(0) + 1.0e-12))
            self.bias.data.copy_(-z.mean(0))
            self.initialized = True

        z = torch.exp(self.log_scale) * z + self.bias
        log_det_jacobians += torch.sum(self.log_scale.view(1, -1), dim=1)
        return z, log_det_jacobians

    def backward(self, y, log_det_jacobians):
        y = (y - self.bias) * torch.exp(-self.log_scale)
        log_det_jacobians -= torch.sum(self.log_scale.view(1, -1), dim=1)
        return y, log_det_jacobians


class BatchNorm(nn.Module):
    def __init__(self, num_features, momentum=0.1, eps=1.0e-5):
        super(BatchNorm, self).__init__()

        self.log_gamma = nn.Parameter(torch.zeros(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.momentum = momentum
        self.eps = eps

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

        self.batch_mean = None
        self.batch_var = None

    def forward(self, x, log_det_jacob):
        if self.training:
            self.batch_mean = x.mean(0)
            self.batch_var = (x - self.batch_mean).pow(2).mean(0) + self.eps

            self.running_mean.mul_(self.momentum)
            self.running_var.mul_(self.momentum)
            self.running_mean.add_(self.batch_mean.detach() * (1.0 - self.momentum))
            self.running_var.add_(self.batch_var.detach() * (1.0 - self.momentum))

            mean, var = self.batch_mean, self.batch_var
        else:
            mean, var = self.running_mean, self.running_var

        x = (x - mean) / torch.sqrt(var)
        x = x * torch.exp(self.log_gamma) + self.beta
        log_det = self.log_gamma - 0.5 * torch.log(var)
        log_det_jacob += torch.sum(log_det)

        return x, log_det_jacob

    def backward(self, x, log_det_jacob):
        if self.training:
            mean, var = self.batch_mean, self.batch_var
        else:
            mean, var = self.running_mean, self.running_var

        x = (x - self.beta) / torch.exp(self.log_gamma)
        x = x * torch.sqrt(var) + mean

        log_det = -self.log_gamma + 0.5 * torch.log(var)
        log_det_jacob += torch.sum(log_det)

        return x, log_det_jacob
