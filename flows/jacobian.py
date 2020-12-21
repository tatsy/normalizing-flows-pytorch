import numpy as np
import torch
import torch.autograd


def trace_df_dz(f, z, method='exact'):
    if method == 'exact':
        trace = _trace_df_dz_exact(f, z)
    elif method == 'approx':
        trace = _trace_df_dz_approx(f, z)
    else:
        raise Exception('unsupported trace extimator: %s ["exact" or "approx" expected]')

    return trace


def _trace_df_dz_exact(f, z):
    """
    matrix trace using native auto-differentiation
    """
    n_dims = z.size(1)
    diags = [torch.autograd.grad(f[:, i].sum(), z, create_graph=True)[0] for i in range(n_dims)]
    diags = [diag[:, i] for i, diag in enumerate(diags)]
    diags = torch.stack(diags, dim=1)
    return torch.sum(diags, dim=1)


def _trace_df_dz_approx(f, z, n_samples=1):
    """
    matrix trace using Hutchinson's stochastic trace estimator
    """
    w_t_J_fn = lambda w, z=z, f=f: torch.autograd.grad(
        f, z, grad_outputs=w, retain_graph=True, create_graph=True)[0]

    w = torch.randn([f.size(0), n_samples, f.size(1)], dtype=torch.float32)
    w = w.to(z.device)

    w_t_J = [w_t_J_fn(w[:, i, :]) for i in range(n_samples)]
    w_t_J = torch.stack(w_t_J, dim=1)

    quad = torch.einsum('bnd,bnd->bn', w_t_J, w)
    sum_diag = torch.mean(quad, dim=1)
    return sum_diag


def logdet_df_dz(g, z, method='fixed'):
    if method == 'fixed':
        logdet = _logdet_df_dz_fixed(g, z)
    elif method == 'unbias':
        logdet = _logdet_df_dz_unbias(g, z)
    else:
        raise Exception('unsupported trace extimator: %s ["fixed" and "unbias" expected]' % method)

    return logdet


def _logdet_df_dz_fixed(g, z, n_samples=1, n_power_series=16):
    """
    log determinant approximation using fixed length cutoff for infinite series
    which can be used with residual block f(x) = x + g(x)
    """

    w_t_J_fn = lambda w, z=z, g=g: torch.autograd.grad(
        g, z, grad_outputs=w, retain_graph=True, create_graph=True)[0]

    v = torch.randn([g.size(0), n_samples, g.size(1)], dtype=torch.float32)
    v = v.to(z.device)

    sum_diag = 0.0
    w = v.clone()
    for k in range(1, n_power_series + 1):
        w = [w_t_J_fn(w[:, i, :]) for i in range(n_samples)]
        w = torch.stack(w, dim=1)

        inner = torch.einsum('bnd,bnd->bn', w, v)
        if (k + 1) % 2 == 0:
            sum_diag += inner / k
        else:
            sum_diag -= inner / k

    sum_diag = torch.mean(sum_diag, dim=1)
    return sum_diag


def _logdet_df_dz_unbias(g, z, n_samples=1, p=0.3):
    """
    log determinant approximation using unbiased series length sampling
    which can be used with residual block f(x) = x + g(x)
    """

    w_t_J_fn = lambda w, z=z, g=g: torch.autograd.grad(
        g, z, grad_outputs=w, retain_graph=True, create_graph=True)[0]

    res = 0.0
    for j in range(n_samples):
        n_iters = np.random.geometric(p)

        v = torch.randn([g.size(0), n_samples, g.size(1)], dtype=torch.float32)
        v = v.to(z.device)
        w = v.clone()

        sum_diag = 0.0
        for k in range(1, n_iters + 1):
            w = [w_t_J_fn(w[:, i, :]) for i in range(n_samples)]
            w = torch.stack(w, dim=1)

            inner = torch.einsum('bnd,bnd->bn', w, v)

            P = (1.0 - p)**(k - 1)
            if (k + 1) % 2 == 0:
                sum_diag += inner / (k * P)
            else:
                sum_diag -= inner / (k * P)

        res += torch.mean(sum_diag, dim=1)

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

            g = g_fn(x)
            ctx.x = x
            ctx.g = g
            logdetJg = logdet_fn(g, x)

            if ctx.training:
                dlogdetJg_dx, *dlogdetJg_dtheta = torch.autograd.grad(logdetJg.sum(), [x] + theta,
                                                                      retain_graph=True,
                                                                      allow_unused=True)
                ctx.save_for_backward(dlogdetJg_dx, *theta, *dlogdetJg_dtheta)

        return safe_detach(g), safe_detach(logdetJg)

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


def safe_detach(x):
    """
    detech operation which keeps reguires_grad
    ---
    https://github.com/rtqichen/residual-flows/blob/master/lib/layers/iresblock.py
    """
    return x.detach().requires_grad_(x.requires_grad)


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

    trace_exact = _trace_df_dz_exact(y, x).item()
    print('det[exact] = %.8f' % (trace_exact))

    trace_hutch = _trace_df_dz_approx(y, x, n_samples=1024).item()
    print('det[approx] = %.8f' % (trace_hutch))
    print('')

    # make spectral norm of operator be less than 1
    print('*** test for log determinant ***')

    eigval, eigvec = np.linalg.eig(AA)
    eigval = eigval / (np.max(np.abs(eigval)) + 1.0)
    AA = np.dot(eigvec, np.dot(np.diag(eigval), eigvec.T))
    print('||A||_2 = %f' % (np.linalg.norm(AA, ord=2)))

    AA = torch.from_numpy(AA)
    II = torch.eye(4)
    f = torch.matmul(x, AA) + bb
    y = x + f

    log_det_real = torch.log(torch.det(II + AA))
    print('log_det[real] = %.8f' % (log_det_real))

    det_res = _logdet_df_dz_fixed(f, x, n_samples=64, n_power_series=16).item()
    print('det[fixed] = %.8f' % (det_res))

    det_res = _logdet_df_dz_unbias(f, x, n_samples=64, p=0.5).item()
    print('det[unbias] = %.8f' % (det_res))
