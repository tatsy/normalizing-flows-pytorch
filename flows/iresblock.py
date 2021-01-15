import numpy as np
import torch
import torch.nn as nn

from .misc import safe_detach
from .modules import LipSwish
from .spectral_norm import SpectralNorm as spectral_norm

activations = {
    'relu': nn.ReLU,
    'elu': nn.ELU,
    'softplus': nn.Softplus,
    'lipswish': LipSwish,
}


def log_df_dz_exact(g, z):
    """
    exact log determinant estimator
    """

    n_dims = z.size(1)

    jac = [
        torch.autograd.grad(g[:, i].sum(), z, create_graph=True, retain_graph=True)[0]
        for i in range(n_dims)
    ]

    jac = torch.stack(jac, dim=1)
    ident = torch.eye(n_dims).type_as(z).to(z.device)
    return torch.logdet(ident + jac)


def log_df_dz_fixed(g, z, n_samples=1, n_power_series=8):
    """
    log determinant approximation using fixed length cutoff for infinite series
    which can be used with residual block f(x) = x + g(x)
    """

    w_t_J_fn = lambda w, z=z, g=g: torch.autograd.grad(
        g, z, grad_outputs=w, retain_graph=True, create_graph=True)[0]

    v = torch.randn([g.size(0), n_samples, g.size(1)])
    v = v.type_as(z).to(z.device)

    sum_diag = 0.0
    w = v.clone()
    for k in range(1, n_power_series + 1):
        w = [w_t_J_fn(w[:, i, :]) for i in range(n_samples)]
        w = torch.stack(w, dim=1)

        inner = torch.einsum('bnd,bnd->bn', w, v)
        sum_diag += (-1)**(k + 1) * (inner / k)

    sum_diag = torch.mean(sum_diag, dim=1)
    return sum_diag


def log_df_dz_unbias(g, z, n_samples=1, p=0.5, n_exact=1, is_training=True):
    """
    log determinant approximation using unbiased series length sampling
    which can be used with residual block f(x) = x + g(x)
    """

    res = 0.0
    for j in range(n_samples):
        n_power_series = n_exact + np.random.geometric(p)

        v = torch.randn_like(g)
        w = v

        sum_vj = 0.0
        for k in range(1, n_power_series + 1):
            w = torch.autograd.grad(g, z, w, create_graph=is_training, retain_graph=True)[0]
            geom_cdf = (1.0 - p)**max(0, (k - n_exact) - 1)
            tr = torch.sum(w * v, dim=1)
            sum_vj = sum_vj + (-1)**(k + 1) * (tr / (k * geom_cdf))

        res += sum_vj

    return res / n_samples


def log_df_dz_neumann(g, z, n_samples=1, p=0.5, n_exact=1):
    """
    log determinant approximation using unbiased series length sampling.
    ---
    NOTE: this method using neumann series does not return exact "log_df_dz"
    but the one that can be only used in gradient wrt parameters.
    """

    res = 0.0
    for j in range(n_samples):
        n_power_series = n_exact + np.random.geometric(p)

        v = torch.randn_like(g)
        w = v

        sum_vj = v
        with torch.no_grad():
            for k in range(1, n_power_series + 1):
                w = torch.autograd.grad(g, z, w, retain_graph=True)[0]
                geom_cdf = (1.0 - p)**max(0, (k - n_exact) - 1)
                sum_vj = sum_vj + ((-1)**k / geom_cdf) * w

        sum_vj = torch.autograd.grad(g, z, sum_vj, create_graph=True)[0]
        res += torch.sum(sum_vj * v, dim=1)

    return res / n_samples


class MemorySavedLogDetEstimator(torch.autograd.Function):
    """
    Memory saving logdet estimator used in Residual Flow
    ---
    This code is borrowed from following URL but revised as it can be understood more easily.
    https://github.com/rtqichen/residual-flows/blob/master/lib/layers/iresblock.py
    """
    @staticmethod
    def forward(ctx, logdet_fn, x, g_fn, training, *g_params):
        ctx.training = training
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            theta = list(g_params)

            # log-det for neumann series
            g = g_fn(x)
            ctx.x = x
            ctx.g = g
            logdetJg = log_df_dz_neumann(g, x)

            if ctx.training:
                dlogdetJg_dx, *dlogdetJg_dtheta = torch.autograd.grad(logdetJg.sum(), [x] + theta,
                                                                      retain_graph=True,
                                                                      allow_unused=True)
                ctx.save_for_backward(dlogdetJg_dx, *theta, *dlogdetJg_dtheta)

            # log-det for loss calculation
            logdet = logdet_fn(g, x)

        return safe_detach(g), safe_detach(logdet)

    @staticmethod
    def backward(ctx, dL_dg, dL_dlogdetJg):
        """
        NOTE: Be careful that chain rule for partial differentiation is as follows

        df(y, z)    df   dy     df   dz
        -------- =  -- * --  +  -- * --
        dx          dy   dx     dz   dx
        """

        training = ctx.training
        if not training:
            raise ValueError('Provide training=True if using backward.')

        # chain rule for partial differentiation (1st term)
        with torch.enable_grad():
            g, x = ctx.g, ctx.x
            dlogdetJg_dx, *saved_tensors = ctx.saved_tensors
            n_params = len(saved_tensors) // 2
            theta = saved_tensors[:n_params]
            dlogdetJg_dtheta = saved_tensors[n_params:]

            dL_dx_1st, *dL_dtheta_1st = torch.autograd.grad(g, [x] + theta,
                                                            grad_outputs=dL_dg,
                                                            allow_unused=True)

        # chain rule for partial differentiation (2nd term)
        # ---
        # NOTE:
        # dL_dlogdetJg consists of same values for all dimensions (see forward).
        dL_dlogdetJg_scalar = dL_dlogdetJg[0].detach()
        with torch.no_grad():
            dL_dx_2nd = dlogdetJg_dx * dL_dlogdetJg_scalar
            dL_dtheta_2nd = tuple(
                [g * dL_dlogdetJg_scalar if g is not None else None for g in dlogdetJg_dtheta])

        with torch.no_grad():
            dL_dx = dL_dx_1st + dL_dx_2nd
            dL_dtheta = tuple([
                g1 + g2 if g2 is not None else g1 for g1, g2 in zip(dL_dtheta_1st, dL_dtheta_2nd)
            ])

        return (None, dL_dx, None, None) + dL_dtheta


def basic_logdet_wrapper(logdet_fn, x, g_fn, training):
    g = g_fn(x)
    logdetJg = logdet_fn(g, x)
    return x + g, logdetJg


def memory_saved_logdet_wrapper(logdet_fn, x, g_fn, training):
    g_params = list(g_fn.parameters())
    return MemorySavedLogDetEstimator.apply(logdet_fn, x, g_fn, training, *g_params)


class InvertibleResBlockBase(nn.Module):
    """
    invertible residual block
    """
    def __init__(self, coeff=0.97, ftol=1.0e-4, logdet_estimator='unbias'):
        super(InvertibleResBlockBase, self).__init__()

        self.coeff = coeff
        self.ftol = ftol
        self.estimator = logdet_estimator
        self.proc_g_fn = memory_saved_logdet_wrapper
        self.g_fn = nn.Sequential()

    def _get_logdet_estimator(self):
        if self.training:
            # force use unbiased log-det estimator
            logdet_fn = lambda g, z: log_df_dz_unbias(g, z, 1, is_training=self.training)
        else:
            if self.estimator == 'exact':
                logdet_fn = log_df_dz_exact
            elif self.estimator == 'fixed':
                logdet_fn = lambda g, z: log_df_dz_fixed(g, z, n_samples=4, n_power_series=8)
            elif self.estimator == 'unbias':
                logdet_fn = lambda g, z: log_df_dz_unbias(
                    g, z, n_samples=4, n_exact=8, is_training=self.training)
            else:
                raise Exception('Unknown log-det estimator: %s' % (self.estimator))

        return logdet_fn

    def forward(self, x, log_df_dz):
        logdet_fn = self._get_logdet_estimator()
        g, logdet = self.proc_g_fn(logdet_fn, x, self.g_fn, self.training)
        z = x + g
        log_df_dz += logdet
        return z, log_df_dz

    def backward(self, z, log_df_dz):
        n_iters = 100
        x = z.clone()
        logdet_fn = self._get_logdet_estimator()

        with torch.enable_grad():
            x.requires_grad_(True)
            for k in range(n_iters):
                x = safe_detach(x)
                g = self.g_fn(x)
                x, prev_x = z - g, x

                if torch.all(torch.abs(x - prev_x) < self.ftol):
                    break

            x = safe_detach(x)
            g = self.g_fn(x)
            logdet = logdet_fn(g, x)

        return x, log_df_dz - logdet


class InvertibleResLinear(InvertibleResBlockBase):
    def __init__(self,
                 in_features,
                 out_features,
                 base_filters=32,
                 n_layers=2,
                 activation='lipswish',
                 coeff=0.97,
                 ftol=1.0e-4,
                 logdet_estimator='unbias'):
        super(InvertibleResLinear, self).__init__(coeff, ftol, logdet_estimator)

        act_fn = activations[activation]
        hidden_dims = [in_features] + [base_filters] * n_layers + [out_features]
        layers = []
        for i, (in_dims, out_dims) in enumerate(zip(hidden_dims[:-1], hidden_dims[1:])):
            layers.append(spectral_norm(nn.Linear(in_dims, out_dims), coeff=self.coeff))
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
                 coeff=0.97,
                 ftol=1.0e-4,
                 logdet_estimator='unbias'):
        super(InvertibleResConv2d, self).__init__(coeff, ftol, logdet_estimator)

        act_fn = activations[activation]
        hidden_dims = [in_channels] + [base_filters] * n_layers + [out_channels]
        layers = []
        for i, (in_dims, out_dims) in enumerate(zip(hidden_dims[:-1], hidden_dims[1:])):
            layers.append(spectral_norm(nn.Conv2d(in_dims, out_dims, 3, 1, 1), coeff=self.coeff))
            if i != len(hidden_dims) - 2:
                layers.append(act_fn())

        self.g_fn = nn.Sequential(*layers)


if __name__ == '__main__':
    # testing log-det estimator
    AA = np.random.normal(size=(4, 4)).astype('float32')
    AA = np.dot(AA.T, AA)

    # make spectral norm of operator be less than 1
    print('*** test for log determinant ***')

    eigval, eigvec = np.linalg.eig(AA)
    eigval = eigval / (np.max(np.abs(eigval)) + 2.0)
    AA = np.dot(eigvec, np.dot(np.diag(eigval), eigvec.T))
    print('||A||_2 = %f' % (np.linalg.norm(AA, ord=2)))

    AA = torch.from_numpy(AA)
    II = torch.eye(4)
    f = torch.matmul(x, AA) + bb
    y = x + f

    log_det_exact = log_df_dz_exact(f, x)
    print('log_det[exact] = %.8f' % (log_det_exact))

    det_res = log_df_dz_fixed(f, x, n_samples=1000, n_power_series=10).item()
    print('det[fixed] = %.8f' % (det_res))

    det_res = log_df_dz_unbias(f, x, n_samples=1000, p=0.5, n_exact=10).item()
    print('det[unbias] = %.8f' % (det_res))
