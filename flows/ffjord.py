import numpy as np
import torch
import torch.nn as nn
import torch.autograd

from .cnf import CNF
from .modules import ActNorm, Compose


class Ffjord(nn.Module):
    def __init__(self, dims, datatype=None, cfg=None):
        super(Ffjord, self).__init__()

        self.dims = dims
        self.n_layers = cfg.layers
        self.stepsize = cfg.stepsize

        t0 = cfg.t0
        t1 = cfg.t1
        steps = int(np.ceil((t1 - t0) / self.stepsize)) + 1
        times = torch.linspace(t0, t1, steps, dtype=torch.float32)

        layers = []
        if datatype == 'image':
            # for image
            NotImplementedError('Sorry, FFJORD for image generation is not supported!')
        else:
            # for density samples
            for i in range(self.n_layers):
                layers.append(ActNorm(dims))
                layers.append(
                    CNF(dims, times=times, solver_type=cfg.solver, trace_estimator=cfg.trace))

        self.net = Compose(layers)

    def forward(self, z):
        log_df_dz = torch.zeros(z.size(0)).type_as(z).to(z.device)
        return self.net(z, log_df_dz)

    def backward(self, z):
        log_df_dz = torch.zeros(z.size(0)).type_as(z).to(z.device)
        return self.net.backward(z, log_df_dz)
