import torch
import torch.nn as nn

from .modules import Network


class BijectiveCoupling(nn.Module):
    def __init__(self, n_dims):
        super(BijectiveCoupling, self).__init__()

        n_half_dims = n_dims // 2
        self.s_scale = nn.Parameter(torch.ones(n_half_dims,
                                               dtype=torch.float32),
                                    requires_grad=True)
        self.s_shift = nn.Parameter(torch.zeros(n_half_dims,
                                                dtype=torch.float32),
                                    requires_grad=True)
        self.net_t = Network(n_dims - n_half_dims, n_half_dims)
        self.net_s = Network(n_dims - n_half_dims, n_half_dims)

    def forward(self, z0, z1):
        t = self.net_t(z1)
        s = torch.tanh(self.net_s(z1)) * self.s_scale + self.s_shift
        z0 = z0 * torch.exp(s) + t
        return z0, z1, s

    def backward(self, z0, z1):
        t = self.net_t(z1)
        s = torch.tanh(self.net_s(z1)) * self.s_scale + self.s_shift
        z0 = torch.exp(-s) * (z0 - t)
        return z0, z1, -s


class RealNVP(nn.Module):
    def __init__(self, n_dims, n_layers=8):
        super(RealNVP, self).__init__()

        self.n_dims = n_dims
        self.n_layers = n_layers

        indices = torch.arange(n_dims, dtype=torch.long)
        mask = torch.where(indices % 2 == 0, torch.ones(n_dims),
                           torch.zeros(n_dims)).long()
        mask = nn.Parameter(mask, requires_grad=False)

        layers = []
        masks = []
        for i in range(self.n_layers):
            layers.append(BijectiveCoupling(n_dims))
            masks.append(mask if i % 2 == 0 else 1.0 - mask)

        self.layers = nn.ModuleList(layers)
        self.masks = masks

    def forward(self, y):
        log_det_jacobians = torch.zeros_like(y)

        z = y
        for i in range(self.n_layers):
            mask = self.masks[i]
            z0 = z[:, mask != 0]
            z1 = z[:, mask == 0]
            z0, z1, log_det_J = self.layers[i](z0, z1)
            z = torch.zeros_like(z)
            z[:, mask != 0] = z0
            z[:, mask == 0] = z1
            log_det_jacobians[:, mask != 0] += log_det_J

        return z, log_det_jacobians

    def backward(self, z):
        log_det_jacobians = torch.zeros_like(z)

        y = z
        for i in reversed(range(self.n_layers)):
            mask = self.masks[i]
            y0 = y[:, mask != 0]
            y1 = y[:, mask == 0]
            y0, y1, log_det_J = self.layers[i].backward(y0, y1)
            y = torch.zeros_like(y)
            y[:, mask != 0] = y0
            y[:, mask == 0] = y1
            log_det_jacobians[:, mask != 0] += log_det_J

        return y, log_det_jacobians
