import numpy as np
import scipy as sp
import torch
import torch.nn as nn
import scipy.linalg
import torch.nn.functional as F

from .realnvp import BijectiveCoupling


class Actnorm(nn.Module):
    def __init__(self, n_channels):
        super(Actnorm, self).__init__()
        self.n_channels = n_channels
        self.log_scale = nn.Parameter(torch.ones(n_channels))
        self.bias = nn.Parameter(torch.zeros(n_channels))
        self.initialized = False

    def forward(self, z, log_det_jacobians):
        if not self.initialized:
            self.log_scale.data.copy_(-torch.log(x.std(0) + 1.0e-12))
            self.bias.data.copy_(x.mean(0))
            self.initialized = True

        z = torch.exp(self.log_scale) * z + self.bias
        log_det_jacobians += self.log_scale
        return z, log_det_jacobians

    def backward(self, y, log_det_jacobians):
        y = (y - self.bias) * torch.exp(-self.log_scale)
        log_det_jacobians -= self.log_scale
        return y, log_det_jacobians


class InvertibleLinear(nn.Module):
    def __init__(self, in_out_channels):
        super(InvertibleLinear, self).__init__()

        W = np.ndarray((in_out_channels, in_out_channels), dtype='float32')
        W = torch.from_numpy(W)
        nn.init.orthogonal_(W)
        LU, pivots = torch.lu(W)

        P, L, U = torch.lu_unpack(LU, pivots)
        self.P = nn.Parameter(P, requires_grad=False)
        self.L = nn.Parameter(L, requires_grad=True)
        self.U = nn.Parameter(U, requires_grad=True)
        self.I = nn.Parameter(torch.eye(in_out_channels), requires_grad=True)
        self.pivots = nn.Parameter(pivots, requires_grad=False)

        L_mask = np.tril(np.ones((in_out_channels, in_out_channels), dtype='float32'), k=-1)
        U_mask = L_mask.T.copy()
        self.L_mask = nn.Parameter(torch.from_numpy(L_mask), requires_grad=False)
        self.U_mask = nn.Parameter(torch.from_numpy(U_mask), requires_grad=False)

        s = np.diag(U)
        sign_s = np.sign(s)
        log_s = np.log(np.abs(s))
        self.log_s = nn.Parameter(torch.from_numpy(log_s), requires_grad=True)
        self.sign_s = nn.Parameter(torch.from_numpy(sign_s), requires_grad=False)

    def forward(self, z, log_det_jacobians):
        L = self.L * self.L_mask + self.I
        U = self.U * self.U_mask + torch.diag(self.sign_s * torch.exp(self.log_s))
        W = self.P @ L @ U

        return z @ W, log_det_jacobians + self.log_s

    def backward(self, y, log_det_jacobians):
        LU = self.L * self.L_mask + self.U * self.U_mask + torch.diag(self.sign_s * torch.exp(self.log_s))

        return torch.lu_solve(LU, self.pivots), log_det_jacobians - self.log_s


class Glow(nn.Module):
    def __init__(self, n_dims, n_layers=8):
        super(Glow, self).__init__()

        self.n_dims = n_dims
        self.n_layers = n_layers

        indices = torch.arange(n_dims, dtype=torch.long)
        mask = torch.where(indices % 2 == 0, torch.ones(n_dims), torch.zeros(n_dims)).long()
        mask = nn.Parameter(mask, requires_grad=False)

        actnorms = []
        linears = []
        couplings = []

        masks = []
        for i in range(self.n_layers):
            actnorms.append(Actnorm(n_dims))
            linears.append(InvertibleLinear(n_dims))
            couplings.append(BijectiveCoupling(n_dims))
            masks.append(mask if i % 2 == 0 else 1.0 - mask)

        self.actnorms = nn.ModuleList(actnorms)
        self.linears = nn.ModuleList(linears)
        self.couplings = nn.ModuleList(couplings)
        self.masks = masks

    def forward(self, y):
        log_det_jacobians = torch.zeros_like(y)

        z = y
        for i in range(self.n_layers):
            # actnorm
            z, log_det_J = self.actnorms[i](z)
            log_det_jacobians += log_det_J

            # invertible linear
            z, log_det_J = self.linears[i](z)
            log_det_jacobians += log_det_J

            # bijective coupling
            mask = self.masks[i]
            z0 = z[:, mask != 0]
            z1 = z[:, mask == 0]
            z0, z1, log_det_J = self.couplings[i](z0, z1)
            z = torch.zeros_like(z)
            z[:, mask != 0] = z0
            z[:, mask == 0] = z1
            log_det_jacobians[:, mask != 0] += log_det_J

        return z, log_det_jacobians

    def backward(self, z):
        log_det_jacobians = torch.zeros_like(z)

        y = z
        for i in reversed(range(self.n_layers)):
            # bijective coupling
            y, log_det_jacobians = self.couplings[i].backward(y, log_det_jacobians)

            # invertible linear
            y, log_det_jacobians = self.linears[i].backward(y, log_det_jacobians)

            # actnorm
            y, log_det_jacobians = self.actnorms[i].backward(y, log_det_jacobians)

        return y, log_det_jacobians
