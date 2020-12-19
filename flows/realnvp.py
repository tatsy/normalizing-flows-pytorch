import torch
import torch.nn as nn

from .modules import BatchNorm
from .coupling import AffineCoupling


class RealNVP(nn.Module):
    def __init__(self, dims, cfg=None):
        super(RealNVP, self).__init__()

        self.dims = dims
        self.n_layers = cfg.network.layers

        layers = []
        bnorms = []
        for i in range(self.n_layers):
            bnorms.append(BatchNorm(dims))
            layers.append(AffineCoupling(dims, odd=i % 2 != 0))

        self.bnorms = nn.ModuleList(bnorms)
        self.layers = nn.ModuleList(layers)

    def forward(self, y):
        z = y
        log_det_jacobians = torch.zeros(z.size(0), dtype=torch.float32, device=z.device)
        for i in range(self.n_layers):
            z, log_det_jacobians = self.bnorms[i](z, log_det_jacobians)
            z, log_det_jacobians = self.layers[i](z, log_det_jacobians)

        return z, log_det_jacobians

    def backward(self, z):
        y = z
        log_det_jacobians = torch.zeros(y.size(0), dtype=torch.float32, device=y.device)
        for i in reversed(range(self.n_layers)):
            y, log_det_jacobians = self.layers[i].backward(y, log_det_jacobians)
            y, log_det_jacobians = self.bnorms[i].backward(y, log_det_jacobians)

        return y, log_det_jacobians
