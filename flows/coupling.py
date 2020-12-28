import torch
import torch.nn as nn

from .modules import MLP, Logit, ConvNet, MixLogCDF
from .squeeze import Squeeze1d, Squeeze2d


class AbstractCoupling(nn.Module):
    def __init__(self, dims, odd=False):
        super(AbstractCoupling, self).__init__()
        self.dims = dims
        if len(dims) == 1:
            # density
            self.squeeze = Squeeze1d(dims, odd)
        elif len(dims) == 3:
            # image
            self.squeeze = Squeeze2d(dims, odd)
        else:
            raise Exception('unsupported dimensions: %s' % (str(dims)))

    def forward(self, z, log_df_dz):
        z0, z1 = self.squeeze.split(z)
        z0, z1, log_df_dz = self._transform(z0, z1, log_df_dz)
        z = self.squeeze.merge(z0, z1)
        return z, log_df_dz

    def backward(self, y, log_df_dz):
        y0, y1 = self.squeeze.split(y)
        y0, y1, log_df_dz = self._inverse_transform(y0, y1, log_df_dz)
        y = self.squeeze.merge(y0, y1)

        return y, log_df_dz

    def _transform(self, z0, z1, log_df_dz):
        pass

    def _inverse_transform(self, z0, z1, log_df_dz):
        pass


class AdditiveCoupling(AbstractCoupling):
    """ additive coupling used in NICE """
    def __init__(self, dims, odd=False):
        super(AdditiveCoupling, self).__init__(dims, odd)
        if len(dims) == 1:
            in_chs = dims[0] // 2 if not odd else (dims[0] + 1) // 2
            out_chs = dims[0] - in_chs
            self.net_t = MLP(in_chs, out_chs)
        elif len(dims) == 3:
            self.net_t = ConvNet(dims[0] * 2, dims[0] * 2)

    def _transform(self, z0, z1, log_df_dz):
        t = self.net_t(z1)
        z0 = z0 + t

        return z0, z1, log_df_dz

    def _inverse_transform(self, y0, y1, log_df_dz):
        t = self.net_t(y1)
        y0 = y0 - t

        return y0, y1, log_df_dz


class AffineCoupling(AbstractCoupling):
    """ affine coupling used in Real NVP """
    def __init__(self, dims, odd=False):
        super(AffineCoupling, self).__init__(dims, odd)
        if len(dims) == 1:
            in_chs = dims[0] // 2 if not odd else (dims[0] + 1) // 2
            out_chs = dims[0] - in_chs
            self.net_s = MLP(in_chs, out_chs)
            self.net_t = MLP(in_chs, out_chs)
        elif len(dims) == 3:
            self.net_s = ConvNet(dims[0] * 2, dims[0] * 2)
            self.net_t = ConvNet(dims[0] * 2, dims[0] * 2)

    def _transform(self, z0, z1, log_df_dz):
        t = self.net_t(z1)
        s = torch.tanh(self.net_s(z1))
        z0 = z0 * torch.exp(s) + t
        log_df_dz += torch.sum(s.view(z0.size(0), -1), dim=1)

        return z0, z1, log_df_dz

    def _inverse_transform(self, y0, y1, log_df_dz):
        t = self.net_t(y1)
        s = torch.tanh(self.net_s(y1))
        y0 = torch.exp(-s) * (y0 - t)
        log_df_dz -= torch.sum(s.view(y0.size(0), -1), dim=1)

        return y0, y1, log_df_dz


class ContinuousMixtureCoupling(AbstractCoupling):
    """ continuous mixture coupling used in Flow++ """
    def __init__(self, dims, odd=False, n_mixtures=4):
        super(ContinuousMixtureCoupling, self).__init__(dims, odd)
        self.n_mixtures = n_mixtures

        if len(dims) == 1:
            in_chs = dims[0] // 2 if not odd else (dims[0] + 1) // 2
            out_chs = dims[0] - in_chs
            self.net_a = MLP(in_chs, out_chs)
            self.net_b = MLP(in_chs, out_chs)
            self.net_pi = MLP(in_chs, out_chs * self.n_mixtures)
            self.net_mu = MLP(in_chs, out_chs * self.n_mixtures)
            self.net_s = MLP(in_chs, out_chs * self.n_mixtures)
        elif len(dims) == 3:
            self.net_a = ConvNet(dims[0] * 2, dims[0] * 2)
            self.net_b = ConvNet(dims[0] * 2, dims[0] * 2)
            self.net_pi = ConvNet(dims[0] * 2, dims[0] * 2 * self.n_mixtures)
            self.net_mu = ConvNet(dims[0] * 2, dims[0] * 2 * self.n_mixtures)
            self.net_s = ConvNet(dims[0] * 2, dims[0] * 2 * self.n_mixtures)

        self.logit = Logit()
        self.mix_log_cdf = MixLogCDF()

    def _transform(self, z0, z1, log_df_dz):
        B = z0.size(0)
        C = z0.size()[1:]
        a = torch.tanh(self.net_a(z1))
        b = self.net_b(z1)
        pi = torch.softmax(self.net_pi(z1).view(B, self.n_mixtures, *C), dim=1)
        mu = self.net_mu(z1).view(B, self.n_mixtures, *C)
        s = self.net_s(z1).view(B, self.n_mixtures, *C)

        z0, log_df_dz = self.mix_log_cdf(z0, pi, mu, s, log_df_dz)
        z0, log_df_dz = self.logit(z0, log_df_dz)

        z0 = z0 * torch.exp(a) + b
        log_df_dz += torch.sum(a.view(z0.size(0), -1), dim=1)

        return z0, z1, log_df_dz

    def _inverse_transform(self, z0, z1, log_df_dz):
        B = z0.size(0)
        C = z0.size()[1:]
        a = torch.tanh(self.net_a(z1))
        b = self.net_b(z1)
        pi = torch.softmax(self.net_pi(z1).view(B, self.n_mixtures, *C), dim=1)
        mu = self.net_mu(z1).view(B, self.n_mixtures, *C)
        s = self.net_s(z1).view(B, self.n_mixtures, *C)

        z0 = torch.exp(-a) * (z0 - b)
        log_df_dz -= torch.sum(a.view(z0.size(0), -1), dim=1)

        z0, log_df_dz = self.logit.backward(z0, log_df_dz)
        z0, log_df_dz = self.mix_log_cdf.backward(z0, pi, mu, s, log_df_dz)

        return z0, z1, log_df_dz
