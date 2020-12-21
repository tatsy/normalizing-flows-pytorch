from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .jacobian import trace_df_dz


class HyperNetwork(nn.Module):
    """ network for hyper parameters """
    def __init__(self, in_features, out_features, base_filters=8):
        super(HyperNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features, base_filters),
            nn.Tanh(),
            nn.Linear(base_filters, base_filters),
            nn.Tanh(),
            nn.Linear(base_filters, out_features),
        )

    def forward(self, t):
        h = self.net(t)
        return h


class HyperLinear(nn.Module):
    """ fully connected layer for continuous normalizing flow """
    def __init__(self, in_features, out_features, base_filters=8):
        super(HyperLinear, self).__init__()

        self.weight_shape = [out_features, in_features]
        self.bias_shape = [out_features]
        self.n_out_params = np.prod(self.weight_shape) + np.prod(self.bias_shape)
        self.hyper_net = HyperNetwork(1, self.n_out_params)

    def forward(self, x, t):
        params = self.hyper_net(t)
        weight = params[:, :np.prod(self.weight_shape)].view(self.weight_shape)
        bias = params[:, np.prod(self.weight_shape):].view(self.bias_shape)
        return F.linear(x, weight, bias)


class HyperConv2d(nn.Module):
    """ 2D convolution for continuous normalizing flow """
    def __init__(self,
                 in_channels,
                 out_channels,
                 base_filters=8,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 groups=1):
        super(HyperConv2d, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.weight_shape = [out_channels, in_channels // groups, kernel_size, kernel_size]
        self.bias_shape = [out_channels]
        self.n_out_params = np.prod(self.weight_shape) + np.prod(self.bias_shape)
        self.hyper_net = HyperNetwork(1, self.n_out_params)

    def forward(self, x, t):
        params = self.hyper_net(t)
        weight = params[:, :np.prod(self.weight_shape)].view(self.weight_shape)
        bias = params[:, np.prod(self.weight_shape):].view(self.bias_shape)
        return F.conv2d(x,
                        weight,
                        bias,
                        stride=self.stride,
                        padding=self.padding,
                        groups=self.groups)


class CNF(nn.Module):
    """ continuous normalizing flow """
    def __init__(self, dims, base_filters=8, n_layers=4, trace_estimate_method='exact'):
        super(CNF, self).__init__()
        self.trace_estimator_fn = partial(trace_df_dz, method=trace_estimate_method)

        if len(dims) == 1:
            # density
            conv_fn = partial(HyperLinear, base_filters=base_filters)
        elif len(dims) == 3:
            # image
            conv_fn = partial(HyperConv2d, base_filters=base_filters)
        else:
            raise Exception('unsupported target dimension: %s' % (str(dims)))

        hidden_dims = [dims[0]] + [base_filters] * n_layers + [dims[0]]
        layers = []
        for in_dims, out_dims in zip(hidden_dims[:-1], hidden_dims[1:]):
            layers.append(conv_fn(in_dims, out_dims))

        self.layers = nn.ModuleList(layers)

    def forward(self, z, t):
        t = t.view(1, 1)

        with torch.enable_grad():
            z.requires_grad_(True)
            for i, layer in enumerate(self.layers):
                z = layer(z, t)
                if i != len(self.layers) - 1:
                    z = torch.relu(z)

            dz_dt = z
            dlogpz_dt = -self.trace_estimator_fn(dz_dt, z)

        return dz_dt, dlogpz_dt
