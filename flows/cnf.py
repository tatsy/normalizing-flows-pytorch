import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .misc import weights_init_as_nearly_identity
from .odeint import odeint, odeint_adjoint


def trace_df_dz_exact(f, z):
    """
    matrix trace using native auto-differentiation
    """
    n_dims = z.size(1)
    diags = [
        torch.autograd.grad(f[:, i].sum(), z, create_graph=True, retain_graph=True)[0][:, i]
        for i in range(n_dims)
    ]
    return sum(diags)


def trace_df_dz_hutchinson(f, z, n_samples=1, is_training=True):
    """
    matrix trace using Hutchinson's stochastic trace estimator
    """
    w_t_J_fn = lambda w, z=z, f=f: torch.autograd.grad(
        f, z, grad_outputs=w, retain_graph=True, create_graph=is_training)[0]

    w = torch.randn([f.size(0), n_samples, f.size(1)])
    w = w.type_as(z).to(z.device)

    w_t_J = [w_t_J_fn(w[:, i, :]) for i in range(n_samples)]
    w_t_J = torch.stack(w_t_J, dim=1)

    quad = torch.einsum('bnd,bnd->bn', w_t_J, w)
    sum_diag = torch.mean(quad, dim=1)
    return sum_diag


class ConcatLinear(nn.Module):
    """
    fully connected layer for continuous normalizing flow
    """
    def __init__(self, in_features, out_features):
        super(ConcatLinear, self).__init__()
        self.linear = nn.Linear(in_features + 1, out_features)

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1]) * t
        t_and_x = torch.cat([tt, x], dim=1)
        return self.linear(t_and_x)


class ConcatConv2d(nn.Module):
    """
    Conv2d layer for continuous normalizing flow
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(ConcatConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels + 1, out_channels, kernel_size, stride, padding)

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1]) * t
        t_and_x = torch.cat([tt, x], dim=1)
        return self.conv(t_and_x)


class ODENet(nn.Module):
    """
    network to define ordinary differential equation
    """
    def __init__(self, dims, base_filters=32, n_layers=2, trace_estimator='hutchinson'):
        super(ODENet, self).__init__()
        self.estimator = trace_estimator

        if len(dims) == 1:
            # density
            conv_fn = ConcatLinear
        elif len(dims) == 3:
            # image
            conv_fn = ConcatConv2d
        else:
            raise Exception('unsupported target dimension: %s' % (str(dims)))

        hidden_dims = [dims[0]] + [base_filters] * n_layers + [dims[0]]
        layers = []
        for in_dims, out_dims in zip(hidden_dims[:-1], hidden_dims[1:]):
            layers.append(conv_fn(in_dims, out_dims))

        self.layers = nn.ModuleList(layers)

    def _get_trace_estimator(self):
        trace_fn = None
        if self.training:
            # force using Hutchinson trace estimator
            trace_fn = lambda f, z: trace_df_dz_hutchinson(
                f, z, n_samples=1, is_training=self.training)
        else:
            if self.estimator == 'exact':
                trace_fn = trace_df_dz_exact
            elif self.estimator == 'hutchinson':
                trace_fn = lambda f, z: trace_df_dz_hutchinson(
                    f, z, n_samples=4, is_training=self.training)

        return trace_fn

    def forward(self, t, states):
        t = t.view(1, 1)
        z = states[0]
        trace_fn = self._get_trace_estimator()
        with torch.enable_grad():
            z.requires_grad_(True)
            dz_dt = z.clone()
            for i, layer in enumerate(self.layers):
                dz_dt = layer(t, dz_dt)
                if i != len(self.layers) - 1:
                    dz_dt = F.softplus(dz_dt)

            dlogpz_dt = trace_fn(dz_dt, z)

        return dz_dt, dlogpz_dt


class CNF(nn.Module):
    """
    continuous normalizing flow
    """
    def __init__(self,
                 dims,
                 times,
                 solver_type,
                 trace_estimator='hutchinson',
                 backprop='adjoint',
                 dtype=torch.float64):
        super(CNF, self).__init__()
        assert backprop in ['normal', 'adjoint'], 'unsupported backprop type "%s"' % (backprop)

        self.dims = dims
        self.dtype = dtype
        self.func = ODENet(dims, trace_estimator=trace_estimator).type(self.dtype)
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


if __name__ == '__main__':
    # testing trace estimator
    AA = np.random.normal(size=(4, 4)).astype('float32')
    AA = np.dot(AA.T, AA)

    # test for ordinary matrix
    print('*** test for matrix trace ***')

    x = torch.randn((1, 4))
    x.requires_grad_(True)
    AA = torch.from_numpy(AA)
    bb = torch.randn((1, 4))
    y = torch.matmul(x, AA) + bb

    trace_real = torch.trace(AA)
    print('det[real] = %.8f' % (trace_real))

    trace_exact = trace_df_dz_exact(y, x).item()
    print('det[exact] = %.8f' % (trace_exact))

    trace_hutch = trace_df_dz_hutchinson(y, x, n_samples=1024).item()
    print('det[hutch] = %.8f' % (trace_hutch))
    print('')
