import torch
import torch.nn as nn

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
    x = x.unsqueeze(-1)
    x = pi * torch.sigmoid((x - mu) * torch.exp(-s))
    x = torch.sum(x, dim=-1)
    return x


def deriv_mix_log_cdf(x, pi, mu, s):
    x = x.unsqueeze(-1)
    w = torch.exp(-s)
    x = pi * deriv_sigmoid((x - mu) * w) * w
    x = torch.sum(x, dim=-1)
    return x


class Sigmoid(nn.Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x, log_det_jacobians):
        log_det = torch.sum(torch.log(deriv_sigmoid(x)), dim=1)
        return torch.sigmoid(x), log_det_jacobians + log_det

    def backward(self, x, log_det_jacobians):
        x = torch.clamp(x, 1.0e-8, 1.0 - 1.0e-8)
        log_det = torch.sum(torch.log(deriv_logit(x)), dim=1)
        return torch.logit(x), log_det_jacobians + log_det


class Logit(nn.Module):
    def __init__(self):
        super(Logit, self).__init__()

    def forward(self, x, log_det_jacobians):
        x = torch.clamp(x, 1.0e-8, 1.0 - 1.0e-8)
        log_det = torch.sum(torch.log(deriv_logit(x)), dim=1)
        return torch.logit(x), log_det_jacobians + log_det

    def backward(self, x, log_det_jacobians):
        log_det = torch.sum(torch.log(deriv_sigmoid(x)), dim=1)
        return torch.sigmoid(x), log_det_jacobians + log_det


class Tanh(nn.Module):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x, log_det_jacobians):
        log_det = torch.sum(torch.log(deriv_tanh(x)), dim=1)
        return torch.tanh(x), log_det_jacobians + log_det

    def backward(self, x, log_det_jacobians):
        log_det = torch.sum(torch.log(deriv_arctanh(x)), dim=1)
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
        log_det = torch.sum(torch.log(deriv_mix_log_cdf(x, pi, mu, s)), dim=1)
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
        log_det = torch.sum(torch.log(deriv_mix_log_cdf(x, pi, mu, s)), dim=1)

        return x, log_det_jacobians - log_det


class LinearWN(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super(LinearWN, self).__init__()
        self.conv = nn.utils.weight_norm(nn.Linear(in_dim, out_dim, bias=bias))

    def forward(self, x):
        return self.conv(x)


class LinearSN(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super(LinearSN, self).__init__()
        self.conv = SpectralNorm(nn.Linear(in_dim, out_dim, bias=bias))

    def forward(self, x):
        return self.conv(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()

        self.net = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            LinearWN(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            LinearWN(out_channels, out_channels),
        )

        if in_channels != out_channels:
            self.bridge = nn.Sequential(LinearWN(in_channels, out_channels))
        else:
            self.bridge = nn.Sequential()

    def forward(self, x):
        y = self.net(x)
        x = self.bridge(x)
        return x + y


class Network(nn.Module):
    def __init__(self, in_channels, out_channels, base_filters=32):
        super(Network, self).__init__()

        self.in_block = nn.Sequential(
            LinearWN(in_channels, base_filters),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True),
            LinearWN(base_filters, base_filters),
        )

        self.mid_block = nn.Sequential(
            ResBlock(base_filters, base_filters),
            ResBlock(base_filters, base_filters),
        )

        self.out_block = nn.Sequential(
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True),
            LinearWN(base_filters, out_channels),
        )

    def forward(self, x):
        x = self.in_block(x)
        x = self.mid_block(x)
        return self.out_block(x)
