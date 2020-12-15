import torch
import torch.nn as nn
import torch.autograd

from .glow import Actnorm
from .modules import SpectralNormLinear


class InvResBlock(nn.Module):
    def __init__(self, in_out_channels):
        super(InvResBlock, self).__init__()
        self.linear = nn.Sequential(
            SpectralNormLinear(in_out_channels, in_out_channels),
            nn.ELU(inplace=True),
            SpectralNormLinear(in_out_channels, in_out_channels),
            nn.ELU(inplace=True),
            SpectralNormLinear(in_out_channels, in_out_channels),
        )

    def forward(self, y, log_det_jacobians):
        n_iters = 16
        n_samples = 1

        B, C = y.size()
        g_y = self.linear(y)

        dims = [B, n_samples, g_y.size(1)]
        v = torch.randn(dims).to(y.device)

        log_det_J = 0.0
        w = v.clone()
        for k in range(1, n_iters + 1):
            new_w = [
                torch.autograd.grad(g_y, y, grad_outputs=w[:, i, :], retain_graph=True, create_graph=True)[0]
                for i in range(n_samples)
            ]
            w = torch.stack(new_w, dim=1)

            inner = torch.einsum('bnd,bnd->bn', w, v)
            if (k + 1) % 2 == 0:
                log_det_J += inner / k
            else:
                log_det_J -= inner / k

        z = y + g_y
        log_det_jacobians = log_det_jacobians + torch.mean(log_det_J, dim=1)
        return z, log_det_jacobians

    def backward(self, z, log_det_jacobians):
        n_iters = 32
        y = z.clone()
        for k in range(n_iters):
            g_y = self.linear(y)
            y = z - g_y

        log_det_J = torch.zeros_like(log_det_jacobians)
        _, log_det_J = self.forward(y, log_det_J)

        return y, log_det_jacobians - log_det_J


class InvResNet(nn.Module):
    def __init__(self, n_dims, n_layers=8):
        super(InvResNet, self).__init__()

        self.n_dims = n_dims
        self.n_layers = n_layers

        actnorms = []
        layers = []
        for i in range(self.n_layers):
            actnorms.append(Actnorm(n_dims))
            layers.append(InvResBlock(n_dims))

        self.actnorms = nn.ModuleList(actnorms)
        self.layers = nn.ModuleList(layers)

    def forward(self, y):
        z = y
        log_det_jacobians = torch.zeros_like(y[:, 0])
        for i in range(self.n_layers):
            z, log_det_jacobians = self.actnorms[i](z, log_det_jacobians)
            z, log_det_jacobians = self.layers[i](z, log_det_jacobians)

        return z, log_det_jacobians

    def backward(self, z):
        y = z
        log_det_jacobians = torch.zeros_like(z[:, 0])
        for i in reversed(range(self.n_layers)):
            y, log_det_jacobians = self.layers[i].backward(y, log_det_jacobians)
            y, log_det_jacobians = self.actnorms[i].backward(y, log_det_jacobians)

        return y, log_det_jacobians
