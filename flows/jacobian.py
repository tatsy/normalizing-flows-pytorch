import numpy as np
import torch
import torch.autograd


def trace_df_dz(f, z, method='exact'):
    if method == 'hutchinson':
        trace = _trace_df_dz_hutchinson(f, z)
    elif method == 'exact':
        trace = _trace_df_dz_exact(f, z)
    else:
        # logger.warning('unsupported trace extimator: %s\n'
        #                'use exact jacobian calculation' % (method))
        trace = _trace_df_dz_exact(f, z)

    return trace


def _trace_df_dz_exact(f, z):
    n_dims = z.size(1)
    diags = [torch.autograd.grad(f[:, i].sum(), z, create_graph=True)[0] for i in range(n_dims)]
    diags = [diag[:, i] for i, diag in enumerate(diags)]
    diags = torch.stack(diags, dim=1)
    return torch.sum(diags, dim=1)


def _trace_df_dz_hutchinson(f, z, n_samples=1):
    w_t_J = lambda w, z=z, f=f: torch.autograd.grad(
        f, z, grad_outputs=w, retain_graph=True, create_graph=True)[0]

    v = torch.randn([f.size(0), n_samples, f.size(1)], dtype=torch.float32)
    w = v.to(z.device)

    w = [w_t_J(w[:, i, :]) for i in range(n_samples)]
    w = torch.stack(w, dim=1)

    inner = torch.einsum('bnd,bnd->bn', w, v)
    sum_diag = torch.mean(inner, dim=1)
    return sum_diag


def logdet_df_dz(g, z, n_iters=16, n_samples=1):
    """ log determinant for residual block with f(x) = x + g(x) """

    w_t_J = lambda w, z=z, g=g: torch.autograd.grad(
        g, z, grad_outputs=w, retain_graph=True, create_graph=True)[0]

    v = torch.randn([g.size(0), n_samples, g.size(1)], dtype=torch.float32)
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


def logdet_df_dz_unbias(g, z, p=0.3, n_samples=1):
    """ log determinant for residual block with f(x) = x + g(x) """

    w_t_J = lambda w, z=z, g=g: torch.autograd.grad(
        g, z, grad_outputs=w, retain_graph=True, create_graph=True)[0]

    res = 0.0
    for j in range(n_samples):
        n_iters = np.random.geometric(p)

        v = torch.randn([g.size(0), n_samples, g.size(1)], dtype=torch.float32)
        v = v.to(z.device)
        w = v.clone()

        sum_diag = 0.0
        for k in range(1, n_iters + 1):
            w = [w_t_J(w[:, i, :]) for i in range(n_samples)]
            w = torch.stack(w, dim=1)

            inner = torch.einsum('bnd,bnd->bn', w, v)

            P = (1.0 - p)**(k - 1)
            if (k + 1) % 2 == 0:
                sum_diag += inner / (k * P)
            else:
                sum_diag -= inner / (k * P)

        res += torch.mean(sum_diag, dim=1)

    return res / n_samples


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

    trace_hutch = _trace_df_dz_hutchinson(y, x, n_samples=1024).item()
    print('det[hutch] = %.8f' % (trace_hutch))
    print('')

    # make spectral norm be less than 1
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

    det_res = logdet_df_dz_unbias(f, x, n_samples=64).item()
    print('det[res] = %.8f' % (det_res))
