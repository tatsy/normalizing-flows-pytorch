import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import Compose, BatchNorm


class MADE(nn.Module):
    def __init__(self, in_out_features, num_hidden=2, base_filters=32, use_companion=False):
        super(MADE, self).__init__()
        self.in_out_chs = in_out_features
        self.num_hidden = num_hidden
        self.base_filters = base_filters
        self.use_companion = use_companion
        self.masks = None

        weights = []
        biases = []
        units = []
        bnorms = []

        hidden_dims = [in_out_features] + [base_filters] * num_hidden
        for in_dims, out_dims in zip(hidden_dims[:-1], hidden_dims[1:]):
            xavier_scale = np.sqrt(2.0 / (out_dims + in_dims))
            W = torch.randn(out_dims, in_dims) * xavier_scale
            U = torch.randn(out_dims, in_dims) * xavier_scale
            b = torch.randn(out_dims) * 0.01
            weights.append(nn.Parameter(W))
            units.append(nn.Parameter(U))
            biases.append(nn.Parameter(b))
            bnorms.append(nn.BatchNorm1d(out_dims))

        xavier_scale = np.sqrt(2.0 / (in_out_features + hidden_dims[-1]))
        W = torch.randn(in_out_features, hidden_dims[-1]) * xavier_scale
        U = torch.randn(in_out_features, hidden_dims[-1]) * xavier_scale
        b = torch.randn(in_out_features) * 0.01
        weights.append(nn.Parameter(W))
        units.append(nn.Parameter(U))
        biases.append(nn.Parameter(b))

        self.weights = nn.ParameterList(weights)
        self.bnorms = nn.ModuleList(bnorms)
        self.biases = nn.ParameterList(biases)

        if use_companion:
            self.units = nn.ParameterList(units)

    def forward(self, z):
        self._create_masks()

        for i in range(self.num_hidden):
            self.masks[i] = self.masks[i].type_as(z).to(z.device)
            h = F.linear(z, self.weights[i] * self.masks[i], self.biases[i])
            if self.use_companion:
                h += F.linear(torch.ones_like(z), self.units[i] * self.masks[i])
            z = torch.relu(self.bnorms[i](h))

        self.masks[-1] = self.masks[-1].type_as(z).to(z.device)
        h = F.linear(z, self.weights[-1] * self.masks[-1], self.biases[-1])
        if self.use_companion:
            h += F.linear(torch.ones_like(z), self.units[-1] * self.masks[-1])

        return h

    def _create_masks(self):
        m_prev = torch.arange(self.in_out_chs)
        hidden_dims = [self.in_out_chs] + [self.base_filters] * self.num_hidden
        masks = []
        for in_dims, out_dims in zip(hidden_dims[:-1], hidden_dims[1:]):
            min_k = min(m_prev.min().item(), self.in_out_chs - 2)
            m = torch.from_numpy(np.random.randint(min_k, self.in_out_chs - 1, size=(out_dims)))
            M = torch.zeros(out_dims, in_dims)
            for k in range(out_dims):
                M[k, :] = (m_prev <= m[k]).float()
            masks.append(M)
            m_prev = m

        M = torch.zeros(self.in_out_chs, hidden_dims[-1])
        m = m_prev
        for k in range(hidden_dims[-1]):
            M[m[k] + 1:, k] = 1.0
        masks.append(M)

        self.masks = masks


class AutoregressiveTransfrom(nn.Module):
    def __init__(self, in_out_features, num_hidden=3, base_filters=32):
        super(AutoregressiveTransfrom, self).__init__()
        self.in_out_chs = in_out_features

        perm = torch.eye(in_out_features)[:, torch.randperm(in_out_features)]
        self.register_buffer('perm', perm)

        self.net_s = MADE(in_out_features, num_hidden, base_filters)
        self.net_t = MADE(in_out_features, num_hidden, base_filters)
        self.register_parameter('s_log_scale', nn.Parameter(torch.randn(1) * 0.01))
        self.register_parameter('s_bias', nn.Parameter(torch.randn(1) * 0.01))

    def forward(self, z, log_df_dz):
        z = torch.mm(z, self.perm)
        s = torch.tanh(self.net_s(z)) * self.s_log_scale + self.s_bias
        t = self.net_t(z)
        z = z * torch.exp(s) + t
        log_df_dz += torch.sum(s, dim=1)
        return z, log_df_dz

    def backward(self, z, log_df_dz):
        m = torch.zeros(self.in_out_chs).type_as(z).to(z.device)
        for i in range(self.in_out_chs):
            s = torch.tanh(self.net_s(z)) * self.s_log_scale + self.s_bias
            t = self.net_t(z)
            z[:, i] = ((z - t) * torch.exp(-s))[:, i]
            log_df_dz -= s[:, i]
            m[i] = 1.0

        z = torch.mm(z, self.perm.t())
        return z, log_df_dz


class MAF(nn.Module):
    def __init__(self, dims, datatype=None, cfg=None):
        super(MAF, self).__init__()

        self.dims = dims
        self.n_layers = cfg.layers

        layers = []
        if datatype == 'image':
            # for image
            NotImplementedError('Sorry, MAF for image generation is not supported!')

        else:
            # for density samples
            for i in range(self.n_layers):
                layers.append(BatchNorm(dims, affine=False))
                layers.append(AutoregressiveTransfrom(dims[0]))

        self.net = Compose(layers)

    def forward(self, z):
        log_df_dz = torch.zeros(z.size(0)).type_as(z).to(z.device)
        return self.net(z, log_df_dz)

    def backward(self, z):
        log_df_dz = torch.zeros(z.size(0)).type_as(z).to(z.device)
        return self.net.backward(z, log_df_dz)
