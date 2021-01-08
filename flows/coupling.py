from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import MLP, Logit, ConvNet, GatedAttn, MixLogCDF, GatedConv2d, GatedLinear
from .squeeze import (squeeze1d, unsqueeze1d, channel_merge, channel_split, checker_merge,
                      checker_split)


class AbstractCoupling(nn.Module):
    """
    abstract class for bijective coupling layers
    """
    def __init__(self, dims, masking='checkerboard', odd=False):
        super(AbstractCoupling, self).__init__()
        self.dims = dims
        if len(dims) == 1:
            self.squeeze = lambda z, odd=odd: squeeze1d(z, odd)
            self.unsqueeze = lambda z0, z1, odd=odd: unsqueeze1d(z0, z1, odd)
        elif len(dims) == 3 and masking == 'checkerboard':
            self.squeeze = lambda z, odd=odd: checker_split(z, odd)
            self.unsqueeze = lambda z0, z1, odd=odd: checker_merge(z0, z1, odd)
        elif len(dims) == 3 and masking == 'channelwise':
            self.squeeze = lambda z, odd=odd: channel_split(z, dim=1, odd=odd)
            self.unsqueeze = lambda z0, z1, odd=odd: channel_merge(z0, z1, dim=1, odd=odd)
        else:
            raise Exception('unsupported combination of masking and dimension: %s, %s' %
                            (masking, str(dims)))

    def forward(self, z, log_df_dz):
        z0, z1 = self.squeeze(z)
        z0, z1, log_df_dz = self._transform(z0, z1, log_df_dz)
        z = self.unsqueeze(z0, z1)
        return z, log_df_dz

    def backward(self, y, log_df_dz):
        y0, y1 = self.squeeze(y)
        y0, y1, log_df_dz = self._inverse_transform(y0, y1, log_df_dz)
        y = self.unsqueeze(y0, y1)

        return y, log_df_dz

    def _transform(self, z0, z1, log_df_dz):
        pass

    def _inverse_transform(self, z0, z1, log_df_dz):
        pass


class AdditiveCoupling(AbstractCoupling):
    """
    additive coupling used in NICE
    """
    def __init__(self, dims, masking='checkerboard', odd=False):
        super(AdditiveCoupling, self).__init__(dims, masking, odd)
        if len(dims) == 1:
            in_chs = dims[0] // 2 if not odd else (dims[0] + 1) // 2
            out_chs = dims[0] - in_chs
            self.net_t = MLP(in_chs, out_chs)
        elif len(dims) == 3:
            if masking == 'checkerboard':
                in_out_chs = dims[0]
            elif masking == 'channelwise':
                in_out_chs = dims[0] // 2
            self.net_t = ConvNet(in_out_chs, in_out_chs)

    def _transform(self, z0, z1, log_df_dz):
        t = self.net_t(z1)
        z0 = z0 + t

        return z0, z1, log_df_dz

    def _inverse_transform(self, y0, y1, log_df_dz):
        t = self.net_t(y1)
        y0 = y0 - t

        return y0, y1, log_df_dz


class AffineCoupling(AbstractCoupling):
    """
    affine coupling used in Real NVP
    """
    def __init__(self, dims, masking='checkerboard', odd=False):
        super(AffineCoupling, self).__init__(dims, masking, odd)

        self.register_parameter('s_log_scale', nn.Parameter(torch.randn(1) * 0.01))
        self.register_parameter('s_bias', nn.Parameter(torch.randn(1) * 0.01))

        if len(dims) == 1:
            in_chs = dims[0] // 2 if not odd else (dims[0] + 1) // 2
            self.out_chs = dims[0] - in_chs
            self.net = MLP(in_chs, self.out_chs * 2)
        elif len(dims) == 3:
            if masking == 'checkerboard':
                in_out_chs = dims[0] * 2
            elif masking == 'channelwise':
                in_out_chs = dims[0] // 2
            self.out_chs = in_out_chs
            self.net = ConvNet(in_out_chs, in_out_chs * 2)

    def _transform(self, z0, z1, log_df_dz):
        params = self.net(z1)
        t = params[:, :self.out_chs]
        s = torch.tanh(params[:, self.out_chs:]) * self.s_log_scale + self.s_bias

        z0 = z0 * torch.exp(s) + t
        log_df_dz += torch.sum(s.view(z0.size(0), -1), dim=1)

        return z0, z1, log_df_dz

    def _inverse_transform(self, y0, y1, log_df_dz):
        params = self.net(y1)
        t = params[:, :self.out_chs]
        s = torch.tanh(params[:, self.out_chs:]) * self.s_log_scale + self.s_bias

        y0 = torch.exp(-s) * (y0 - t)
        log_df_dz -= torch.sum(s.view(y0.size(0), -1), dim=1)

        return y0, y1, log_df_dz


class MixLogAttnCoupling(AbstractCoupling):
    """
    mixture logistic coupling with attention used in Flow++
    """
    def __init__(self, dims, masking='checkerboard', odd=False, base_filters=32, n_mixtures=4):
        super(MixLogAttnCoupling, self).__init__(dims, masking, odd)
        self.n_mixtures = n_mixtures

        self.register_parameter('a_log_scale', nn.Parameter(torch.randn(1) * 0.01))
        self.register_parameter('a_bias', nn.Parameter(torch.randn(1) * 0.01))

        if len(dims) == 1:
            in_chs = dims[0] // 2 if not odd else (dims[0] + 1) // 2
            out_chs = dims[0] - in_chs
            mid_shape = (base_filters, ) + tuple(d // 2 for d in dims[1:])
            self.sections = [out_chs] * 2 + [out_chs * self.n_mixtures] * 3

            self.net = nn.Sequential(
                nn.Linear(in_chs, base_filters),
                GatedLinear(base_filters, base_filters),
                nn.LayerNorm(mid_shape),
                GatedAttn(mid_shape, base_filters),
                nn.LayerNorm(mid_shape),
                nn.Linear(base_filters, sum(self.sections)),
            )
        elif len(dims) == 3:
            if masking == 'checkerboard':
                in_chs = dims[0] * 2
                mid_shape = (base_filters, ) + tuple(d // 2 for d in dims[1:])
                self.sections = [dims[0] * 2] * 2 + [dims[0] * 2 * self.n_mixtures] * 3
            elif masking == 'channelwise':
                in_chs = dims[0] // 2
                mid_shape = (base_filters, ) + dims[1:]
                self.sections = [dims[0] // 2] * 2 + [dims[0] // 2 * self.n_mixtures] * 3

            self.net = nn.Sequential(
                nn.Conv2d(in_chs, base_filters, 3, 1, 1),
                GatedConv2d(base_filters, base_filters),
                nn.LayerNorm(mid_shape),
                GatedAttn(mid_shape, base_filters),
                nn.LayerNorm(mid_shape),
                nn.Conv2d(base_filters, sum(self.sections), 3, 1, 1),
            )

        self.logit = Logit()
        self.mix_log_cdf = MixLogCDF()

    def _transform(self, z0, z1, log_df_dz):
        B = z0.size(0)
        C = z0.size()[1:]

        params = self.net(z1)
        a, b, logpi, mu, s = torch.split(params, self.sections, dim=1)
        a = torch.tanh(a) * self.a_log_scale + self.a_bias

        logpi = F.log_softmax(logpi.view(B, self.n_mixtures, *C), dim=1)
        mu = mu.view(B, self.n_mixtures, *C)
        s = s.view(B, self.n_mixtures, *C)

        z0, log_df_dz = self.mix_log_cdf(z0, logpi, mu, s, log_df_dz)
        z0, log_df_dz = self.logit(z0, log_df_dz)

        z0 = z0 * torch.exp(a) + b
        log_df_dz += torch.sum(a.view(z0.size(0), -1), dim=1)

        return z0, z1, log_df_dz

    def _inverse_transform(self, z0, z1, log_df_dz):
        B = z0.size(0)
        C = z0.size()[1:]

        params = self.net(z1)
        a, b, logpi, mu, s = torch.split(params, self.sections, dim=1)
        a = torch.tanh(a) * self.a_log_scale + self.a_bias

        logpi = F.log_softmax(logpi.view(B, self.n_mixtures, *C), dim=1)
        mu = mu.view(B, self.n_mixtures, *C)
        s = s.view(B, self.n_mixtures, *C)

        z0 = torch.exp(-a) * (z0 - b)
        log_df_dz -= torch.sum(a.view(z0.size(0), -1), dim=1)

        z0, log_df_dz = self.logit.backward(z0, log_df_dz)
        z0, log_df_dz = self.mix_log_cdf.backward(z0, logpi, mu, s, log_df_dz)

        return z0, z1, log_df_dz
