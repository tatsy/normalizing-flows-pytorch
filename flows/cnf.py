from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .odeint import odeint_adjoint as odeint
# from .odeint import odeint
from .jacobian import trace_df_dz


def weights_init(m):
    """
    initialize weights as HyperXXXX layers work as identity by default
    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1 or classname.find('Conv') != -1:
        nn.init.constant_(m.weight, 0.0)
        nn.init.normal_(m.bias, 0, 0.01)


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
        self.hyper_net.apply(weights_init)

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
        self.hyper_net.apply(weights_init)

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


class ODENet(nn.Module):
    """ continuous normalizing flow """
    def __init__(self, dims, base_filters=8, n_layers=4, trace_estimate_method='exact'):
        super(ODENet, self).__init__()
        self.trace_estimator_fn = partial(trace_df_dz, method=trace_estimate_method)

        if len(dims) == 1:
            # density
            conv_fn = HyperLinear
        elif len(dims) == 3:
            # image
            conv_fn = HyperConv2d
        else:
            raise Exception('unsupported target dimension: %s' % (str(dims)))

        hidden_dims = [dims[0]] + [base_filters] * n_layers + [dims[0]]
        layers = []
        for in_dims, out_dims in zip(hidden_dims[:-1], hidden_dims[1:]):
            layers.append(conv_fn(in_dims, out_dims, base_filters=base_filters))

        self.layers = nn.ModuleList(layers)

    def forward(self, t, states):
        t = t.view(1, 1)
        z = states[0]
        with torch.enable_grad():
            z.requires_grad_(True)
            t.requires_grad_(True)

            dz_dt = z
            for i, layer in enumerate(self.layers):
                dz_dt = layer(dz_dt, t)
                if i != len(self.layers) - 1:
                    dz_dt = F.softplus(dz_dt)

            dlogpz_dt = self.trace_estimator_fn(dz_dt, z)

        return dz_dt, dlogpz_dt


class CNF(nn.Module):
    def __init__(self, dims, times, solver_type, trace_estimate_method):
        super(CNF, self).__init__()

        self.dims = dims
        self.func = ODENet(dims, trace_estimate_method=trace_estimate_method)
        self.method = solver_type
        self.register_buffer('times', times)

    def forward(self, z, log_df_dz):
        times = reversed(self.times)
        states = odeint(self.func, (z, log_df_dz), times, self.method)
        z = states[0]
        log_df_dz = states[1]
        return z, log_df_dz

    def backward(self, z, log_df_dz):
        times = self.times
        states = odeint(self.func, (z, log_df_dz), times, self.method)
        z = states[0]
        log_df_dz = states[1]
        return z, log_df_dz
