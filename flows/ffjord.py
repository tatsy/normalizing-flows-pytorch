from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.autograd

from .cnf import CNF
from .jacobian import trace_df_dz


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


SOLVERS = {
    'midpoint': Midpoint,
    'rk4': RK4,
}


class Ffjord(nn.Module):
    def __init__(self, dims, cfg):
        super(Ffjord, self).__init__()

        self.dims = dims
        self.func = CNF(dims, n_layers=cfg.network.layers, trace_estimate_method=cfg.network.trace)
        self.t0 = cfg.network.t0
        self.t1 = cfg.network.t1
        self.stepsize = cfg.network.stepsize

        steps = int(np.ceil((self.t1 - self.t0) / self.stepsize)) + 1
        times = torch.linspace(self.t0, self.t1, steps, dtype=torch.float32)
        self.register_buffer('times', times)
        self.solver = SOLVERS[cfg.network.solver]()

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
