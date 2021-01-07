import torch
import torch.nn as nn

from .glow import InvertibleConv1x1
from .modules import Logit, ActNorm, Compose
from .squeeze import Squeeze2d, Unsqueeze2d
from .coupling import MixLogAttnCoupling


class Flowpp(nn.Module):
    def __init__(self, dims, datatype=None, cfg=None):
        super(Flowpp, self).__init__()

        self.dims = dims
        self.n_layers = cfg.layers

        layers = []
        if datatype == 'image':
            # for image
            layers.append(Logit(eps=0.01))

            # multi-scale architecture
            mid_dims = dims
            while max(mid_dims[1], mid_dims[2]) > 8:
                # checkerboard masking
                for i in range(self.n_layers):
                    layers.append(ActNorm(mid_dims))
                    layers.append(InvertibleConv1x1(mid_dims[0]))
                    layers.append(
                        MixLogAttnCoupling(mid_dims,
                                           masking='checkerboard',
                                           odd=i % 2 != 0,
                                           n_mixtures=cfg.mixtures))

                # squeeze
                layers.append(Squeeze2d(odd=False))
                mid_dims = (mid_dims[0] * 4, mid_dims[1] // 2, mid_dims[2] // 2)

                # channel-wise masking
                for i in range(self.n_layers):
                    layers.append(ActNorm(mid_dims))
                    layers.append(InvertibleConv1x1(mid_dims[0]))
                    layers.append(
                        MixLogAttnCoupling(mid_dims,
                                           masking='channelwise',
                                           odd=i % 2 != 0,
                                           n_mixtures=cfg.mixtures))

            # checkerboard masking (lowest resolution)
            for i in range(self.n_layers):
                layers.append(ActNorm(mid_dims))
                layers.append(InvertibleConv1x1(mid_dims[0]))
                layers.append(
                    MixLogAttnCoupling(mid_dims,
                                       masking='checkerboard',
                                       odd=i % 2 != 0,
                                       n_mixtures=cfg.mixtures))

            while mid_dims[1] != dims[1] or mid_dims[2] != dims[2]:
                # channel-wise masking
                # for i in range(self.n_layers):
                #     layers.append(ActNorm(mid_dims))
                #     layers.append(
                #         MixLogAttnCoupling(mid_dims,
                #                            masking='channelwise',
                #                            odd=i % 2 != 0,
                #                            n_mixtures=cfg.mixtures))

                # unsqueeze
                layers.append(Unsqueeze2d(odd=False))
                mid_dims = (mid_dims[0] // 4, mid_dims[1] * 2, mid_dims[2] * 2)

                # checkerboard masking
                # for i in range(self.n_layers):
                #     layers.append(ActNorm(mid_dims))
                #     layers.append(
                #         MixLogAttnCoupling(mid_dims,
                #                            masking='checkerboard',
                #                            odd=i % 2 != 0,
                #                            n_mixtures=cfg.mixtures))

        else:
            # for density samples
            for i in range(self.n_layers):
                layers.append(ActNorm(dims))
                layers.append(MixLogAttnCoupling(dims, odd=i % 2 != 0, n_mixtures=cfg.mixtures))

        self.net = Compose(layers)

    def forward(self, z):
        log_df_dz = torch.zeros(z.size(0)).type_as(z).to(z.device)
        return self.net(z, log_df_dz)

    def backward(self, z):
        log_df_dz = torch.zeros(z.size(0)).type_as(z).to(z.device)
        return self.net.backward(z, log_df_dz)
