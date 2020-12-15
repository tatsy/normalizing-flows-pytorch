import torch
import torch.nn as nn

from .modules import Network


class BijectiveCoupling(nn.Module):
    def __init__(self, n_dims, mask):
        super(BijectiveCoupling, self).__init__()

        n_half_dims = n_dims // 2
        self.s_scale = nn.Parameter(torch.ones(n_half_dims, dtype=torch.float32), requires_grad=True)
        self.s_shift = nn.Parameter(torch.zeros(n_half_dims, dtype=torch.float32), requires_grad=True)
        self.net_t = Network(n_dims - n_half_dims, n_half_dims)
        self.net_s = Network(n_dims - n_half_dims, n_half_dims)
        self.mask = nn.Parameter(mask, requires_grad=False)

    def forward(self, z, log_det_jacobians):
        z0 = z[:, self.mask != 0]
        z1 = z[:, self.mask == 0]

        t = self.net_t(z1)
        s = torch.tanh(self.net_s(z1)) * self.s_scale + self.s_shift
        z0 = z0 * torch.exp(s) + t

        z = torch.zeros_like(z)
        z[:, self.mask != 0] = z0
        z[:, self.mask == 0] = z1

        log_det_jacobians += torch.sum(s, dim=1)

        return z, log_det_jacobians

    def backward(self, y, log_det_jacobians):
        y0 = y[:, self.mask != 0]
        y1 = y[:, self.mask == 0]

        t = self.net_t(y1)
        s = torch.tanh(self.net_s(y1)) * self.s_scale + self.s_shift
        y0 = torch.exp(-s) * (y0 - t)

        y = torch.zeros_like(y)
        y[:, self.mask != 0] = y0
        y[:, self.mask == 0] = y1

        log_det_jacobians -= torch.sum(s, dim=1)

        return y, log_det_jacobians


class RealNVP(nn.Module):
    def __init__(self, n_dims, n_layers=8):
        super(RealNVP, self).__init__()

        self.n_dims = n_dims
        self.n_layers = n_layers

        indices = torch.arange(n_dims, dtype=torch.long)
        mask = torch.where(indices % 2 == 0, torch.ones(n_dims), torch.zeros(n_dims)).long()

        layers = []
        for i in range(self.n_layers):
            m = mask if i % 2 == 0 else 1.0 - mask
            layers.append(BijectiveCoupling(n_dims, m))

        self.layers = nn.ModuleList(layers)

    def forward(self, y):
        z = y
        log_det_jacobians = torch.zeros_like(y[:, 0])
        for i in range(self.n_layers):
            z, log_det_jacobians = self.layers[i](z, log_det_jacobians)

        return z, log_det_jacobians

    def backward(self, z):
        y = z
        log_det_jacobians = torch.zeros_like(z[:, 0])
        for i in reversed(range(self.n_layers)):
            y, log_det_jacobians = self.layers[i].backward(y, log_det_jacobians)

        return y, log_det_jacobians
