import numpy as np
import torch
import torch.nn as nn
import torch.autograd

from .jacobian import trace_df_dz


class HyperNetwork(nn.Module):
    """ neural net for hyper parameters """
    def __init__(self, in_out_channels, base_filters=32, width=64):
        super(HyperNetwork, self).__init__()
        self.in_out_channels = in_out_channels
        self.width = width

        self.net = nn.Sequential(
            nn.Linear(1, base_filters),
            nn.Tanh(),
            nn.Linear(base_filters, base_filters),
            nn.Tanh(),
        )

        blocksize = width * in_out_channels
        self.out_W = nn.Linear(base_filters, blocksize)
        self.out_U = nn.Linear(base_filters, blocksize)
        self.out_G = nn.Linear(base_filters, blocksize)
        self.out_B = nn.Linear(base_filters, width)

    def forward(self, t):
        h = self.net(t)
        W = self.out_W(h).view(self.width, self.in_out_channels, 1)
        U = self.out_U(h).view(self.width, 1, self.in_out_channels)
        G = self.out_G(h).view(self.width, 1, self.in_out_channels)
        B = self.out_B(h).view(self.width, 1, 1)
        U = U * torch.sigmoid(G)
        return W, B, U


class CNF(nn.Module):
    """ continuous normalizing flow """
    def __init__(self, in_out_channels, base_filters=32, width=32):
        super(CNF, self).__init__()
        self.width = width
        self.hyp_net = HyperNetwork(in_out_channels, base_filters, width)

    def forward(self, z, t):
        t = t.view(1, 1)

        with torch.set_grad_enabled(True):
            z.requires_grad_(True)
            W, B, U = self.hyp_net(t)
            Z = z.unsqueeze(0).repeat(self.width, 1, 1)
            h = torch.tanh(torch.matmul(Z, W) + B)
            dz_dt = torch.matmul(h, U).mean(0)
            dlogpz_dt = -trace_df_dz(dz_dt, z, method='exact')

        return dz_dt, dlogpz_dt


class Midpoint(object):
    def __init__(self):
        pass

    def __call__(self, x, t, func, dt):
        k1, l1 = func(x, t)
        k2, l2 = func(x + 0.5 * k1 * dt, t + 0.5 * dt)
        return k2 * dt, l2 * dt


class RK4(object):
    def __init__(self):
        pass

    def __call__(self, x, t, func, dt):
        k1, l1 = func(x, t)
        k2, l2 = func(x + 0.5 * k1 * dt, t + 0.5 * dt)
        k3, l3 = func(x + 0.5 * k2 * dt, t + 0.5 * dt)
        k4, l4 = func(x + k3 * dt, t + dt)
        dk = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        dl = (l1 + 2.0 * l2 + 2.0 * l3 + l4) / 6.0
        return dk * dt, dl * dt


class Ffjord(nn.Module):
    def __init__(self, n_dims, t0=0, t1=2, stepsize=1, *args, **kwargs):
        super(Ffjord, self).__init__()

        self.n_dims = n_dims
        self.func = CNF(n_dims)

        steps = int(np.ceil((t1 - t0) / stepsize)) + 1
        times = torch.linspace(t0, t1, steps, dtype=torch.float32)
        self.times = nn.Parameter(times, requires_grad=False)
        self.solver = RK4()

    def forward(self, z1):
        z = z1
        log_det_jacobians = torch.zeros_like(z[:, 0])
        for t0, t1 in zip(reversed(self.times[:-1]), reversed(self.times[1:])):
            dt = t1 - t0
            dz, dlogpz = self.solver(z, t1, self.func, -dt)
            z = z + dz
            log_det_jacobians = log_det_jacobians - dlogpz

        return z, log_det_jacobians

    def backward(self, z0):
        z = z0
        log_det_jacobians = torch.zeros_like(z[:, 0])
        for t0, t1 in zip(self.times[:-1], self.times[1:]):
            dt = t1 - t0
            dz, dlogpz = self.solver(z, t0, self.func, dt)
            z = z + dz
            log_det_jacobians = log_det_jacobians - dlogpz

        return z, log_det_jacobians
