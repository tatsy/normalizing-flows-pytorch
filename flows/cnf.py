from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .odeint import odeint, odeint_adjoint
from .jacobian import trace_df_dz


class ConcatLinear(nn.Module):
    def __init__(self, dim_in, dim_out, base_filters):
        super(ConcatLinear, self).__init__()
        self._layer = nn.Linear(dim_in + 1, dim_out)

    def forward(self, t, x):
        tt = torch.ones_like(x[:, 0:1]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class HyperNetwork(nn.Module):
    """ network for hyper parameters """
    def __init__(self, in_features, out_features, base_filters=8):
        super(HyperNetwork, self).__init__()

        self.net = nn.Sequential(nn.Linear(in_features, out_features))

    def forward(self, t):
        h = self.net(t)
        return h


class HyperLinear(nn.Module):
    """ fully connected layer for continuous normalizing flow """
    def __init__(self, in_features, out_features, base_filters=8):
        super(HyperLinear, self).__init__()

        self.weight_shape = [out_features, in_features]
        self.bias_shape = [out_features]

        self.linear = nn.Linear(in_features, out_features)
        self.n_out_params = np.prod(self.weight_shape) + np.prod(self.bias_shape)
        self.hyper_net = HyperNetwork(1, self.n_out_params)

    def forward(self, t, x):
        params = self.hyper_net(t)
        weight = params[:, :np.prod(self.weight_shape)].view(self.weight_shape)
        bias = params[:, np.prod(self.weight_shape):].view(self.bias_shape)
        scale = F.linear(x, weight, bias)
        return self.linear(x) * torch.sigmoid(scale)


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

        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride,
                              padding,
                              groups=groups)
        self.n_out_params = np.prod(self.weight_shape) + np.prod(self.bias_shape)
        self.hyper_net = HyperNetwork(1, self.n_out_params)

    def forward(self, t, x):
        params = self.hyper_net(t)
        weight = params[:, :np.prod(self.weight_shape)].view(self.weight_shape)
        bias = params[:, np.prod(self.weight_shape):].view(self.bias_shape)
        scale = F.conv2d(x,
                         weight,
                         bias,
                         stride=self.stride,
                         padding=self.padding,
                         groups=self.groups)
        return self.conv(x) * torch.sigmoid(scale)


class ODENet(nn.Module):
    """ continuous normalizing flow """
    def __init__(self, dims, base_filters=8, n_layers=2, trace_estimate_method='exact'):
        super(ODENet, self).__init__()
        self.trace_fn = lambda f, z, method=trace_estimate_method: trace_df_dz(f, z, method)

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
            dz_dt = z.clone()
            for i, layer in enumerate(self.layers):
                dz_dt = layer(t, dz_dt)
                if i != len(self.layers) - 1:
                    dz_dt = F.softplus(dz_dt)

            dlogpz_dt = 1.0 * self.trace_fn(dz_dt, z)

        return dz_dt, dlogpz_dt


class CNF(nn.Module):
    def __init__(self,
                 dims,
                 times,
                 solver_type,
                 trace_estimate_method,
                 backprop='adjoint',
                 dtype=torch.float64):
        super(CNF, self).__init__()
        assert backprop in ['normal', 'adjoint'], 'unsupported backprop type "%s"' % (backprop)

        self.dims = dims
        self.dtype = dtype
        self.func = ODENet(dims, trace_estimate_method=trace_estimate_method).type(self.dtype)
        self.method = solver_type
        self.backprop = backprop
        self.register_buffer('times', times.type(self.dtype))

    def forward(self, z, log_df_dz):
        org_type = z.type()

        z, log_df_dz = z.type(self.dtype), log_df_dz.type(self.dtype)
        times = torch.flip(self.times, dims=[0])

        if self.backprop == 'normal':
            states = odeint(self.func, (z, log_df_dz), times, self.method)
        elif self.backprop == 'adjoint':
            states = odeint_adjoint(self.func, (z, log_df_dz), times, self.method)

        z = states[0].type(org_type)
        log_df_dz = states[1].type(org_type)
        return (z, log_df_dz)

    def backward(self, z, log_df_dz):
        org_type = z.type()

        z, log_df_dz = z.type(self.dtype), log_df_dz.type(self.dtype)
        times = self.times

        if self.backprop == 'normal':
            states = odeint(self.func, (z, log_df_dz), times, self.method)
        elif self.backprop == 'adjoint':
            states = odeint_adjoint(self.func, (z, log_df_dz), times, self.method)

        z = states[0].type(org_type)
        log_df_dz = states[1].type(org_type)
        return (z, log_df_dz)
