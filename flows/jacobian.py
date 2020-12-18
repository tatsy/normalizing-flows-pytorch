import torch
import torch.autograd

from common.logging import Logging

logger = Logging(__file__)


def trace_df_dz(f, z, method='exact'):
    if method == 'hutchinson':
        trace = _trace_df_dz_hutchinson(f, z)
    elif method == 'exact':
        trace = _trace_df_dz_exact(f, z)
    else:
        logger.warning('unsupported trace extimator: %s\n'
                       'use exact jacobian calculation' % (method))
        trace = _trace_df_dz_exact(f, z)

    return trace


def _trace_df_dz_exact(f, z):
    n_dims = z.size(1)
    diags = [torch.autograd.grad(f[:, i].sum(), z, create_graph=True)[0] for i in range(n_dims)]
    diags = [diag[:, i] for i, diag in enumerate(diags)]
    diags = torch.stack(diags, dim=1)
    return torch.sum(diags, dim=1)


def _trace_df_dz_hutchinson(f, z, n_iters=8, n_samples=1):
    w_t_J = lambda w, z=z, f=f: torch.autograd.grad(
        f, z, grad_outputs=w, retain_graph=True, create_graph=True)[0]

    v = torch.randn([f.size(0), n_samples, f.size(1)], dtype=torch.float32)
    v = v.to(z.device)

    sum_diag = 0.0
    w = v.clone()
    for k in range(1, n_iters + 1):
        w = [w_t_J(w[:, i, :]) for i in range(n_samples)]
        w = torch.stack(w, dim=1)

        inner = torch.einsum('bnd,bnd->bn', w, v)
        if (k + 1) % 2 == 0:
            sum_diag += inner / k
        else:
            sum_diag -= inner / k

    sum_diag = torch.mean(sum_diag, dim=1)
    return sum_diag
