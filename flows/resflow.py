import torch
import torch.nn as nn
import torch.autograd

from .modules import ActNorm, Compose
from .iresblock import InvertibleResConv2d, InvertibleResLinear


class ResFlow(nn.Module):
    def __init__(self, dims, datatype=None, cfg=None):
        super(ResFlow, self).__init__()

        self.dims = dims
        self.n_layers = cfg.layers

        layers = []
        if datatype == 'image':
            # for image
            NotImplementedError('Sorry, residual flow for image generation is not supported!')
        else:
            # for density samples
            for i in range(self.n_layers):
                layers.append(ActNorm(dims))
                layers.append(
                    InvertibleResLinear(dims[0],
                                        dims[0],
                                        coeff=cfg.spnorm_coeff,
                                        logdet_estimator=cfg.logdet))

        self.net = Compose(layers)

    def forward(self, z):
        log_df_dz = torch.zeros(z.size(0)).type_as(z).to(z.device)
        return self.net(z, log_df_dz)

    def backward(self, z):
        log_df_dz = torch.zeros(z.size(0)).type_as(z).to(z.device)
        return self.net.backward(z, log_df_dz)
