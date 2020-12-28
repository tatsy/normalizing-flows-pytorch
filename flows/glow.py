import numpy as np
import torch
import torch.nn as nn

from .modules import ActNorm, Identity
from .coupling import AffineCoupling


class InvertibleConv1x1(nn.Module):
    """
    invertible 1x1 convolution used in Glow
    """
    def __init__(self, in_out_channels):
        super(InvertibleConv1x1, self).__init__()

        W = torch.zeros((in_out_channels, in_out_channels), dtype=torch.float32)
        nn.init.orthogonal_(W)
        LU, pivots = torch.lu(W)

        P, L, U = torch.lu_unpack(LU, pivots)
        self.P = nn.Parameter(P, requires_grad=False)
        self.L = nn.Parameter(L, requires_grad=True)
        self.U = nn.Parameter(U, requires_grad=True)
        self.I = nn.Parameter(torch.eye(in_out_channels), requires_grad=False)
        self.pivots = nn.Parameter(pivots, requires_grad=False)

        L_mask = np.tril(np.ones((in_out_channels, in_out_channels), dtype='float32'), k=-1)
        U_mask = L_mask.T.copy()
        self.L_mask = nn.Parameter(torch.from_numpy(L_mask), requires_grad=False)
        self.U_mask = nn.Parameter(torch.from_numpy(U_mask), requires_grad=False)

        s = torch.diag(U)
        sign_s = torch.sign(s)
        log_s = torch.log(torch.abs(s))
        self.log_s = nn.Parameter(log_s, requires_grad=True)
        self.sign_s = nn.Parameter(sign_s, requires_grad=False)

    def forward(self, z, log_df_dz):
        L = self.L * self.L_mask + self.I
        U = self.U * self.U_mask + torch.diag(self.sign_s * torch.exp(self.log_s))
        W = self.P @ L @ U

        B = z.size(0)
        C = z.size(1)
        z = torch.matmul(W, z.view(B, C, -1)).view(z.size())

        num_pixels = np.prod(z.size()) // (z.size(0) * z.size(1))
        log_df_dz += torch.sum(self.log_s, dim=0) * num_pixels

        return z, log_df_dz

    def backward(self, y, log_df_dz):
        with torch.no_grad():
            LU = self.L * self.L_mask + self.U * self.U_mask + torch.diag(
                self.sign_s * torch.exp(self.log_s))

            y_reshape = y.view(y.size(0), y.size(1), -1)
            y_reshape = torch.lu_solve(y_reshape, LU.unsqueeze(0), self.pivots.unsqueeze(0))
            y = y_reshape.view(y.size())

        num_pixels = np.prod(y.size()) // (y.size(0) * y.size(1))
        log_df_dz -= torch.sum(self.log_s, dim=0) * num_pixels

        return y, log_df_dz


class Glow(nn.Module):
    def __init__(self, dims, in_act_fn=None, cfg=None):
        super(Glow, self).__init__()

        self.dims = dims
        self.n_layers = cfg.network.layers

        actnorms = []
        linears = []
        couplings = []
        for i in range(self.n_layers):
            actnorms.append(ActNorm(dims))
            linears.append(InvertibleConv1x1(dims[0]))
            couplings.append(AffineCoupling(dims, odd=i % 2 != 0))

        self.in_act_fn = in_act_fn() if in_act_fn is not None else Identity()
        self.actnorms = nn.ModuleList(actnorms)
        self.linears = nn.ModuleList(linears)
        self.couplings = nn.ModuleList(couplings)

    def forward(self, z):
        log_df_dz = torch.zeros(z.size(0)).type_as(z).to(z.device)
        z, log_df_dz = self.in_act_fn(z, log_df_dz)
        for i in range(self.n_layers):
            # actnorm
            z, log_df_dz = self.actnorms[i](z, log_df_dz)

            # invertible linear
            z, log_df_dz = self.linears[i](z, log_df_dz)

            # bijective coupling
            z, log_df_dz = self.couplings[i](z, log_df_dz)

        return z, log_df_dz

    def backward(self, z):
        log_df_dz = torch.zeros(z.size(0)).type_as(z).to(z.device)
        for i in reversed(range(self.n_layers)):
            # bijective coupling
            z, log_df_dz = self.couplings[i].backward(z, log_df_dz)

            # invertible linear
            z, log_df_dz = self.linears[i].backward(z, log_df_dz)

            # actnorm
            z, log_df_dz = self.actnorms[i].backward(z, log_df_dz)

        z, log_df_dz = self.in_act_fn.backward(z, log_df_dz)
        return z, log_df_dz
