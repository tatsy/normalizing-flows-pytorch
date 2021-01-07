import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import Compose, BatchNorm, deriv_tanh


class PlanarTransform(nn.Module):
    def __init__(self, dim):
        super(PlanarTransform, self).__init__()

        self.dim = dim
        u = torch.randn(1, self.dim) * 0.01
        w = torch.randn(1, self.dim) * 0.01
        b = torch.randn(1) * 0.01

        self.register_parameter('u', nn.Parameter(u))
        self.register_parameter('w', nn.Parameter(w))
        self.register_parameter('b', nn.Parameter(b))
        self._make_invertible()

    def _make_invertible(self):
        u = self.u
        w = self.w
        w_dot_u = torch.mm(u, w.t())
        if w_dot_u.item() >= -1.0:
            return

        norm_w = w / torch.norm(w, p=2, dim=1)**2
        bias = -1.0 + F.softplus(w_dot_u)
        u = u + (bias - w_dot_u) * norm_w
        self.u.data = u

    def forward(self, z, log_df_dz):
        self._make_invertible()

        w_dot_u = torch.mm(self.u, self.w.t())
        affine = torch.mm(z, self.w.t()) + self.b

        z = z + self.u * torch.tanh(affine)
        det = 1.0 + w_dot_u * deriv_tanh(affine)
        log_df_dz = log_df_dz + torch.sum(torch.log(torch.abs(det) + 1.0e-5), dim=1)

        return z, log_df_dz

    def backward(self, z, log_df_dz):
        w_dot_z = torch.mm(z, self.w.t())
        w_dot_u = torch.mm(self.u, self.w.t())

        lo = torch.full_like(w_dot_z, -1.0e3)
        hi = torch.full_like(w_dot_z, 1.0e3)

        for _ in range(100):
            mid = (lo + hi) * 0.5
            val = mid + w_dot_u * torch.tanh(mid + self.b)
            lo = torch.where(val < w_dot_z, mid, lo)
            hi = torch.where(val > w_dot_z, mid, hi)

            if torch.all(torch.abs(hi - lo) < 1.0e-5):
                break

        affine = (lo + hi) * 0.5 + self.b
        z = z - self.u * torch.tanh(affine)
        det = 1.0 + w_dot_u * deriv_tanh(affine)
        log_df_dz = log_df_dz - torch.sum(torch.log(torch.abs(det) + 1.0e-5), dim=1)

        return z, log_df_dz


class PlanarFlow(nn.Module):
    def __init__(self, dims, datatype=None, cfg=None):
        super(PlanarFlow, self).__init__()

        self.dims = dims
        self.dim = np.prod(dims)
        self.n_layers = cfg.layers

        layers = []
        for i in range(self.n_layers):
            BatchNorm(self.dims)
            layers.append(PlanarTransform(self.dim))

        self.net = Compose(layers)

    def forward(self, z):
        log_df_dz = torch.zeros(z.size(0)).type_as(z).to(z.device)
        return self.net(z, log_df_dz)

    def backward(self, z):
        log_df_dz = torch.zeros(z.size(0)).type_as(z).to(z.device)
        return self.net.backward(z, log_df_dz)
