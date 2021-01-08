import torch
import torch.nn as nn

from .modules import Logit, Compose, BatchNorm
from .squeeze import Squeeze2d, Unsqueeze2d
from .coupling import AffineCoupling, AdditiveCoupling


class RealNVP(nn.Module):
    def __init__(self, dims, datatype=None, cfg=None):
        super(RealNVP, self).__init__()

        self.dims = dims
        self.n_layers = cfg.layers

        layers = []
        if datatype == 'image':
            # for image data
            layers.append(Logit(eps=0.01))

            # multi-scale architecture
            mid_dims = dims
            while max(mid_dims[1], mid_dims[2]) > 8:
                # checkerboard masking
                for i in range(self.n_layers):
                    layers.append(BatchNorm(mid_dims, affine=False))
                    layers.append(AffineCoupling(mid_dims, masking='checkerboard', odd=i % 2 != 0))

                # squeeze
                layers.append(Squeeze2d(odd=False))
                mid_dims = (mid_dims[0] * 4, mid_dims[1] // 2, mid_dims[2] // 2)

                # channel-wise masking
                for i in range(self.n_layers):
                    layers.append(BatchNorm(mid_dims, affine=False))
                    layers.append(AffineCoupling(mid_dims, masking='channelwise', odd=i % 2 != 0))

            # checkerboard masking (lowest resolution)
            for i in range(self.n_layers + 1):
                layers.append(BatchNorm(mid_dims, affine=False))
                layers.append(AffineCoupling(mid_dims, masking='checkerboard', odd=i % 2 != 0))

            # restore to original scale
            while mid_dims[1] != dims[1] or mid_dims[2] != dims[2]:
                # unsqueeze
                layers.append(Unsqueeze2d(odd=False))
                mid_dims = (mid_dims[0] // 4, mid_dims[1] * 2, mid_dims[2] * 2)

        else:
            # for density samples
            for i in range(self.n_layers):
                layers.append(BatchNorm(dims, affine=False))
                layers.append(AffineCoupling(dims, odd=i % 2 != 0))

        self.net = Compose(layers)

    def forward(self, z):
        log_df_dz = torch.zeros(z.size(0)).type_as(z).to(z.device)
        return self.net(z, log_df_dz)

    def backward(self, z):
        log_df_dz = torch.zeros(z.size(0)).type_as(z).to(z.device)
        return self.net.backward(z, log_df_dz)
