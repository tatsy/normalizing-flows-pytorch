import torch
import torch.nn as nn

from .modules import Logit, Network, MixLogCDF


class BijectiveCoupling(nn.Module):
    def __init__(self, n_dims, mask, n_mix=4):
        super(BijectiveCoupling, self).__init__()
        n_half_dims = torch.sum(mask).long().item()

        self.n_mix = n_mix
        self.a_scale = nn.Parameter(torch.ones(n_half_dims, dtype=torch.float32),
                                    requires_grad=True)
        self.a_shift = nn.Parameter(torch.zeros(n_half_dims, dtype=torch.float32),
                                    requires_grad=True)
        self.net_a = Network(n_dims - n_half_dims, n_half_dims)
        self.net_b = Network(n_dims - n_half_dims, n_half_dims)
        self.net_pi = Network(n_dims - n_half_dims, n_half_dims * n_mix)
        self.net_mu = Network(n_dims - n_half_dims, n_half_dims * n_mix)
        self.net_s = Network(n_dims - n_half_dims, n_half_dims * n_mix)
        self.mask = mask

        self.logit = Logit()
        self.mix_log_cdf = MixLogCDF()

    def forward(self, z, log_det_jacobians):
        z0 = z[:, self.mask != 0]
        z1 = z[:, self.mask == 0]
        z0, z1, log_det_jacobians = self._affine_forward(z0, z1, log_det_jacobians)

        z = torch.zeros_like(z)
        z[:, self.mask != 0] = z0
        z[:, self.mask == 0] = z1

        return z, log_det_jacobians

    def _affine_forward(self, z0, z1, log_det_jacobians):
        B, C0 = z0.size()
        a = torch.tanh(self.net_a(z1)) * self.a_scale + self.a_shift
        b = self.net_b(z1)
        pi = torch.softmax(self.net_pi(z1).view(B, C0, self.n_mix), dim=-1)
        mu = self.net_mu(z1).view(B, C0, self.n_mix)
        s = self.net_s(z1).view(B, C0, self.n_mix)

        z0, log_det_jacobians = self.mix_log_cdf(z0, pi, mu, s, log_det_jacobians)
        z0, log_det_jacobians = self.logit(z0, log_det_jacobians)

        z0 = z0 * torch.exp(a) + b
        log_det_jacobians += torch.sum(a, dim=1)

        return z0, z1, log_det_jacobians

    def backward(self, y, log_det_jacobians):
        y0 = y[:, self.mask != 0]
        y1 = y[:, self.mask == 0]
        y0, y1, log_det_jacobians = self._affine_backward(y0, y1, log_det_jacobians)

        y = torch.zeros_like(y)
        y[:, self.mask != 0] = y0
        y[:, self.mask == 0] = y1

        return y, log_det_jacobians

    def _affine_backward(self, z0, z1, log_det_jacobians):
        B, C0 = z0.size()
        a = torch.tanh(self.net_a(z1)) * self.a_scale + self.a_shift
        b = self.net_b(z1)
        pi = torch.softmax(self.net_pi(z1).view(B, C0, self.n_mix), dim=-1)
        mu = self.net_mu(z1).view(B, C0, self.n_mix)
        s = self.net_s(z1).view(B, C0, self.n_mix)

        z0 = torch.exp(-a) * (z0 - b)
        log_det_jacobians -= torch.sum(a, dim=1)

        z0, log_det_jacobians = self.logit.backward(z0, log_det_jacobians)
        z0, log_det_jacobians = self.mix_log_cdf.backward(z0, pi, mu, s, log_det_jacobians)

        return z0, z1, log_det_jacobians


class Flowxx(nn.Module):
    def __init__(self, n_dims, n_layers=4, n_mix=4):
        super(Flowxx, self).__init__()

        self.n_dims = n_dims
        self.n_layers = n_layers

        indices = torch.arange(n_dims, dtype=torch.long)
        mask = torch.where(indices % 2 == 0, torch.ones(n_dims), torch.zeros(n_dims)).long()
        mask = nn.Parameter(mask, requires_grad=False)

        layers = []
        for i in range(self.n_layers):
            m = mask if i % 2 == 0 else 1 - mask
            layers.append(BijectiveCoupling(n_dims, m, n_mix))

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
