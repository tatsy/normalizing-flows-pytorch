import torch
import torch.nn as nn
import torch.autograd

from .modules import ActNorm, Identity
from .iresblock import InvertibleResBlock


class ResFlow(nn.Module):
    def __init__(self, dims, in_act_fn=None, cfg=None):
        super(ResFlow, self).__init__()

        self.dims = dims
        self.n_layers = cfg.network.layers

        actnorms = []
        layers = []
        for i in range(self.n_layers):
            actnorms.append(ActNorm(dims))
            layers.append(InvertibleResBlock(dims[0], logdet_estimate_method=cfg.network.logdet))

        self.in_act_fn = in_act_fn() if in_act_fn is not None else Identity()
        self.actnorms = nn.ModuleList(actnorms)
        self.layers = nn.ModuleList(layers)

    def forward(self, z):
        log_df_dz = torch.zeros(z.size(0)).type_as(z).to(z.device)
        z, log_df_dz = self.in_act_fn(z, log_df_dz)
        for i in range(self.n_layers):
            z, log_df_dz = self.actnorms[i](z, log_df_dz)
            z, log_df_dz = self.layers[i](z, log_df_dz)

        return z, log_df_dz

    def backward(self, z):
        log_df_dz = torch.zeros(z.size(0)).type_as(z).to(z.device)
        for i in reversed(range(self.n_layers)):
            z, log_df_dz = self.layers[i].backward(z, log_df_dz)
            z, log_df_dz = self.actnorms[i].backward(z, log_df_dz)
        z, log_df_dz = self.in_act_fn.backward(z, log_df_dz)

        return z, log_df_dz
