from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.autograd

from .cnf import CNF
from .modules import Identity, BatchNorm


class Ffjord(nn.Module):
    def __init__(self, dims, in_act_fn=None, cfg=None):
        super(Ffjord, self).__init__()

        self.dims = dims
        self.n_layers = cfg.network.layers
        self.stepsize = cfg.network.stepsize

        t0 = cfg.network.t0
        t1 = cfg.network.t1
        steps = int(np.ceil((t1 - t0) / self.stepsize)) + 1
        times = torch.linspace(t0, t1, steps, dtype=torch.float32)

        layers = []
        bnorms = []
        for i in range(self.n_layers):
            layers.append(
                CNF(dims,
                    times=times,
                    solver_type=cfg.network.solver,
                    trace_estimate_method=cfg.network.trace))
            bnorms.append(BatchNorm(dims))

        self.in_act_fn = in_act_fn() if in_act_fn is not None else Identity()
        self.layers = nn.ModuleList(layers)
        self.bnorms = nn.ModuleList(bnorms)

    def forward(self, z1):
        z = z1
        log_df_dz = torch.zeros(z.size(0)).type_as(z).to(z.device)
        z, log_df_dz = self.in_act_fn(z, log_df_dz)
        for i in range(self.n_layers):
            z, log_df_dz = self.bnorms[i](z, log_df_dz)
            z, log_df_dz = self.layers[i](z, log_df_dz)

        return z, log_df_dz

    def backward(self, z0):
        z = z0
        log_df_dz = torch.zeros(z.size(0)).type_as(z).to(z.device)
        for i in reversed(range(self.n_layers)):
            z, log_df_dz = self.layers[i].backward(z, log_df_dz)
            z, log_df_dz = self.bnorms[i].backward(z, log_df_dz)
        z, log_df_dz = self.in_act_fn.backward(z, log_df_dz)

        return z, log_df_dz
