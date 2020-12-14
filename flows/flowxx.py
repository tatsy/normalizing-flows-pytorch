import torch
import torch.nn as nn

from .modules import Network, deriv_logit, deriv_sigmoid


def mix_log_cdf(x, pi, mu, s):
    x = pi * torch.sigmoid((x - mu) * torch.exp(-s))
    x = torch.sum(x, dim=1, keepdim=True)
    return x


def deriv_mix_log_cdf(x, pi, mu, s):
    w = torch.exp(-s)
    x = pi * deriv_sigmoid((x - mu) * w) * w
    x = torch.sum(x, dim=1, keepdim=True)
    return x


class BijectiveCoupling(nn.Module):
    def __init__(self, n_dims, n_mix=4):
        super(BijectiveCoupling, self).__init__()
        n_half_dims = n_dims // 2
        self.n_mix = n_mix
        self.a_scale = nn.Parameter(torch.ones(n_half_dims,
                                               dtype=torch.float32),
                                    requires_grad=True)
        self.a_shift = nn.Parameter(torch.zeros(n_half_dims,
                                                dtype=torch.float32),
                                    requires_grad=True)
        self.net_a = Network(n_dims - n_half_dims, n_half_dims)
        self.net_b = Network(n_dims - n_half_dims, n_half_dims)
        self.net_pi = Network(n_dims - n_half_dims, n_half_dims * n_mix)
        self.net_mu = Network(n_dims - n_half_dims, n_half_dims * n_mix)
        self.net_s = Network(n_dims - n_half_dims, n_half_dims * n_mix)

    def forward(self, z0, z1):
        a = torch.tanh(self.net_a(z1)) * self.a_scale + self.a_shift
        b = self.net_b(z1)
        pi = torch.softmax(self.net_pi(z1).view(-1, self.n_mix), dim=1)
        mu = self.net_mu(z1).view(-1, self.n_mix)
        s = self.net_s(z1).view(-1, self.n_mix)

        log_det_jacobian = torch.zeros_like(z0)
        log_det_jacobian += torch.log(deriv_mix_log_cdf(z0, pi, mu, s))
        z0 = mix_log_cdf(z0, pi, mu, s)

        log_det_jacobian += torch.log(deriv_logit(z0))
        z0 = torch.logit(z0)

        log_det_jacobian += a
        z0 = z0 * torch.exp(a) + b

        return z0, z1, log_det_jacobian

    def backward(self, z0, z1):
        a = torch.tanh(self.net_a(z1)) * self.a_scale + self.a_shift
        b = self.net_b(z1)
        pi = torch.softmax(self.net_pi(z1).view(-1, self.n_mix), dim=1)
        mu = self.net_mu(z1).view(-1, self.n_mix)
        s = self.net_s(z1).view(-1, self.n_mix)

        log_det_jacobian = torch.zeros_like(z0)
        log_det_jacobian += -a
        z0 = torch.exp(-a) * (z0 - b)

        log_det_jacobian += torch.log(deriv_sigmoid(z0))
        z0 = torch.sigmoid(z0)

        lo = torch.full_like(z0, -1.0e3)
        hi = torch.full_like(z0, 1.0e3)
        for it in range(100):
            mid = (lo + hi) * 0.5
            val = mix_log_cdf(mid, pi, mu, s)
            lo = torch.where(val < z0, mid, lo)
            hi = torch.where(val > z0, mid, hi)

            if torch.abs(hi - lo).max() < 1.0e-5:
                break

        z0 = (lo + hi) * 0.5
        log_det_jacobian += -1.0 * torch.log(deriv_mix_log_cdf(z0, pi, mu, s))

        return z0, z1, log_det_jacobian


class FlowXX(nn.Module):
    def __init__(self, n_dims, n_layers=8):
        super(FlowXX, self).__init__()

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
