from functools import partial

import torch
import torch.nn as nn

from .modules import LipSwish
from .jacobian import logdet_df_dz, basic_logdet_wrapper, memory_saved_logdet_wrapper
from .spectral_norm import SpectralNorm as spectral_norm

activations = {
    'relu': nn.ReLU,
    'elu': nn.ELU,
    'softplus': nn.Softplus,
    'lipswish': LipSwish,
}


class InvertibleResBlockBase(nn.Module):
    """ invertible residual block """
    def __init__(self, ftol=1.0e-4, logdet_estimate_method='unbias'):
        super(InvertibleResBlockBase, self).__init__()

        self.ftol = ftol
        self.logdet_fn = partial(logdet_df_dz, method=logdet_estimate_method)
        self.proc_g_fn = memory_saved_logdet_wrapper
        self.g_fn = nn.Sequential()

    def forward(self, x, log_det_jacobians):
        g, logdet = self.proc_g_fn(self.logdet_fn, x, self.g_fn, self.training)

        z = x + g
        log_det_jacobians += logdet
        return z, log_det_jacobians

    def backward(self, z, log_det_jacobians):
        n_iters = 100
        x = z.clone()

        with torch.enable_grad():
            x.requires_grad_(True)
            for k in range(n_iters):
                g = self.g_fn(x)
                x, prev_x = z - g, x

                if torch.all(torch.abs(x - prev_x) < self.ftol):
                    break

            logdet = torch.zeros_like(log_det_jacobians)
            _, logdet = self.forward(x, logdet)

        return x, log_det_jacobians - logdet


class InvertibleResLinear(InvertibleResBlockBase):
    def __init__(self,
                 in_features,
                 out_features,
                 base_filters=32,
                 n_layers=2,
                 activation='lipswish',
                 ftol=1.0e-4,
                 logdet_estimate_method='unbias'):
        super(InvertibleResLinear, self).__init__(ftol, logdet_estimate_method)

        act_fn = activations[activation]
        hidden_dims = [in_features] + [base_filters] * n_layers + [out_features]
        layers = []
        for i, (in_dims, out_dims) in enumerate(zip(hidden_dims[:-1], hidden_dims[1:])):
            layers.append(spectral_norm(nn.Linear(in_dims, out_dims)))
            if i != len(hidden_dims) - 2:
                layers.append(act_fn())

        self.g_fn = nn.Sequential(*layers)


class InvertibleResConv2d(InvertibleResBlockBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 base_filters=32,
                 n_layers=2,
                 activation='lipswish',
                 ftol=1.0e-4,
                 logdet_estimate_method='unbias'):
        super(InvertibleResConv2d, self).__init__(ftol, logdet_estimate_method)

        act_fn = activations[activation]
        hidden_dims = [in_channels] + [base_filters] * n_layers + [out_channels]
        layers = []
        for i, (in_dims, out_dims) in enumerate(zip(hidden_dims[:-1], hidden_dims[1:])):
            layers.append(spectral_norm(nn.Conv2d(in_dims, out_dims, 3, 1, 1)))
            if i != len(hidden_dims) - 2:
                layers.append(act_fn())

        self.g_fn = nn.Sequential(*layers)
