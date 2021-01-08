import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# NOTE:
# weight normalization in Pytorch (i.e., torch.nn.utils.weight_norm)
# does not support using "eps" to disable zero-division.
from .weight_norm import WeightNorm
# NOTE:
# spectral normalization below is that used in iResNet,
# which is different from the original one [Miyato et al. 2018]
# (in Pytorch, it is provided by nn.utils.spectral_norm)
# where it normalize the spectral norm when it is larger than
# the value specfied by "coeff" (0.97 by default).
from .spectral_norm import SpectralNorm


def log_deriv_sigmoid(x):
    """ logarithm of sigmoid derivative """
    return x - 2.0 * F.softplus(x)


def deriv_sigmoid(x):
    """ derivative of sigmoid """
    return torch.exp(log_deriv_sigmoid(x))


def log_deriv_logit(x, eps=1.0e-8):
    """ logarithm of logit derivative """
    y = torch.logit(torch.clamp(x, eps, 1.0 - eps))
    return -log_deriv_sigmoid(y)


def deriv_logit(x, eps=1.0e-8):
    """ derivative of logit """
    return torch.exp(log_deriv_logit(x, eps))


def deriv_tanh(x):
    """ derivative of tanh """
    y = torch.tanh(x)
    return 1.0 - y * y


def log_cosh(x):
    """ numerically stable log cosh(x) """
    s = torch.abs(x)
    p = torch.exp(-2.0 * s)
    return s + torch.log1p(p) - np.log(2.0)


def log_deriv_tanh(x):
    """ logarithm of derivative of tanh """
    return -2.0 * log_cosh(x)


def deriv_arctanh(x, eps=1.0e-8):
    """ derivative of arctanh """
    x = torch.clamp(x, -1.0 + eps, 1.0 - eps)
    return 1.0 / (1.0 - x * x)


def logistic_logpdf(x, mu, s):
    """ logarithm of logistic function """
    z = (x - mu) * torch.exp(-s)
    return z - s - 2.0 * F.softplus(z)


def logistic_logcdf(x, mu, s):
    """ logarithm of logistic CDF """
    z = (x - mu) * torch.exp(-s)
    return F.logsigmoid(z)


def mix_logistic_logpdf(x, logpi, mu, s):
    """
    logarithm of mixture of logistic PDF
    ---
    NOTE: for numerical stability PDF should be computed in
    logarithmic scale using "logsumexp".
    """
    x = x.unsqueeze(1)
    logsig = logistic_logpdf(x, mu, s)
    return torch.logsumexp(logpi + logsig, dim=1)


def mix_logistic_logcdf(x, logpi, mu, s):
    """
    logarithm of mixture of logistic CDF
    ---
    NOTE: for numerical stability PDF should be computed in
    logarithmic scale using "logsumexp".
    """
    x = x.unsqueeze(1)
    logsig = logistic_logcdf(x, mu, s)
    return torch.logsumexp(logpi + logsig, dim=1)


def weight_norm_wrapper(module, wrap=True):
    if wrap:
        return WeightNorm(module)
    else:
        return module


def spectral_norm_wrapper(module, wrap=True):
    if wrap:
        return SpectralNorm(module)
    else:
        return module


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x, log_df_dz):
        return x, log_df_dz

    def backward(self, x, log_df_dz):
        return x, log_df_dz


class Sigmoid(nn.Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x, log_df_dz):
        log_det = log_deriv_sigmoid(x)
        log_det = torch.sum(log_det.view(x.size(0), -1), dim=1)
        return torch.sigmoid(x), log_df_dz + log_det

    def backward(self, x, log_df_dz):
        x = torch.clamp(x, 1.0e-8, 1.0 - 1.0e-8)
        log_det = log_deriv_logit(x)
        log_det = torch.sum(log_det.view(x.size(0), -1), dim=1)
        return torch.logit(x), log_df_dz + log_det


class Logit(nn.Module):
    def __init__(self, eps=1.0e-5):
        super(Logit, self).__init__()
        self.eps = eps

    def forward(self, x, log_df_dz):
        x = torch.clamp(x, self.eps, 1.0 - self.eps)
        log_det = log_deriv_logit(x)
        log_det = torch.sum(log_det.view(x.size(0), -1), dim=1)
        return torch.logit(x), log_df_dz + log_det

    def backward(self, x, log_df_dz):
        log_det = log_deriv_sigmoid(x)
        log_det = torch.sum(log_det.view(x.size(0), -1), dim=1)
        return torch.sigmoid(x), log_df_dz + log_det


class Tanh(nn.Module):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x, log_df_dz):
        log_det = torch.log(deriv_tanh(x))
        log_det = torch.sum(log_det.view(x.size(0), -1), dim=1)
        return torch.tanh(x), log_df_dz + log_det

    def backward(self, x, log_df_dz):
        log_det = torch.log(deriv_arctanh(x))
        log_det = torch.sum(log_det.view(x.size(0), -1), dim=1)
        return torch.arctanh(x), log_df_dz + log_det


class Arctanh(nn.Module):
    def __init__(self):
        super(Arctanh, self).__init__()

    def forward(self, x, log_df_dz):
        log_det = torch.sum(torch.log(deriv_arctanh(x)), dim=1)
        return torch.arctanh(x), log_df_dz + log_det

    def backward(self, x, log_df_dz):
        log_det = torch.sum(torch.log(deriv_tanh(x)), dim=1)
        return torch.tanh(x), log_df_dz + log_det


class MixLogCDF(nn.Module):
    def __init__(self):
        super(MixLogCDF, self).__init__()

    def forward(self, x, log_pi, mu, s, log_df_dz):
        log_det = mix_logistic_logpdf(x, log_pi, mu, s)
        log_det = torch.sum(log_det.view(x.size(0), -1), dim=1)
        logz = mix_logistic_logcdf(x, log_pi, mu, s)
        return torch.exp(logz), log_df_dz + log_det

    def backward(self, x, log_pi, mu, s, log_df_dz):
        lo = torch.full_like(x, -1.0e3)
        hi = torch.full_like(x, 1.0e3)
        for _ in range(100):
            mid = (lo + hi) * 0.5
            val = torch.exp(mix_logistic_logcdf(mid, log_pi, mu, s))
            lo = torch.where(val < x, mid, lo)
            hi = torch.where(val > x, mid, hi)

            if torch.all(torch.abs(hi - lo) < 1.0e-4):
                break

        x = (lo + hi) * 0.5
        log_det = mix_logistic_logpdf(x, log_pi, mu, s)
        log_det = torch.sum(log_det.view(x.size(0), -1), dim=1)

        return x, log_df_dz - log_det


class LipSwish(nn.Module):
    def __init__(self):
        super(LipSwish, self).__init__()
        beta = nn.Parameter(torch.ones([1], dtype=torch.float32))
        self.register_parameter('beta', beta)

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x) / 1.1


class ActNorm(nn.Module):
    def __init__(self, num_features, eps=1.0e-5):
        super(ActNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps

        self.dimensions = [1] + [1 for _ in num_features]
        self.dimensions[1] = num_features[0]
        self.register_parameter('log_scale', nn.Parameter(torch.zeros(self.dimensions)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(self.dimensions)))
        self.initialized = False

    def forward(self, z, log_df_dz):
        if not self.initialized:
            z_reshape = z.view(z.size(0), self.num_features[0], -1)
            log_std = torch.log(torch.std(z_reshape, dim=[0, 2]) + self.eps)
            mean = torch.mean(z_reshape, dim=[0, 2])
            self.log_scale.data.copy_(log_std.view(self.dimensions))
            self.bias.data.copy_(mean.view(self.dimensions))
            self.initialized = True

        z = (z - self.bias) / torch.exp(self.log_scale)

        num_pixels = np.prod(z.size()) // (z.size(0) * z.size(1))
        log_df_dz -= torch.sum(self.log_scale) * num_pixels
        return z, log_df_dz

    def backward(self, y, log_df_dz):
        y = y * torch.exp(self.log_scale) + self.bias
        num_pixels = np.prod(y.size()) // (y.size(0) * y.size(1))
        log_df_dz += torch.sum(self.log_scale) * num_pixels
        return y, log_df_dz


class BatchNorm(nn.Module):
    def __init__(self, num_features, momentum=0.1, eps=1.0e-5, affine=True):
        super(BatchNorm, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.dimensions = [1] + [1 for _ in num_features]
        self.dimensions[1] = num_features[0]
        log_gamma = torch.zeros(self.dimensions)
        beta = torch.zeros(self.dimensions)
        if affine:
            self.register_parameter('log_gamma', nn.Parameter(log_gamma))
            self.register_parameter('beta', nn.Parameter(beta))
        else:
            self.register_buffer('log_gamma', log_gamma)
            self.register_buffer('beta', beta)

        self.register_buffer('running_mean', torch.zeros(self.dimensions))
        self.register_buffer('running_var', torch.ones(self.dimensions))
        self.register_buffer('batch_mean', torch.zeros(self.dimensions))
        self.register_buffer('batch_var', torch.ones(self.dimensions))

    def forward(self, x, log_det_jacob):
        if self.training:
            x_reshape = x.view(x.size(0), self.num_features[0], -1)
            x_mean = torch.mean(x_reshape, dim=[0, 2], keepdim=True)
            x_var = torch.mean((x_reshape - x_mean).pow(2), dim=[0, 2], keepdim=True) + self.eps
            self.batch_mean.data.copy_(x_mean.view(self.dimensions))
            self.batch_var.data.copy_(x_var.view(self.dimensions))

            self.running_mean.mul_(1.0 - self.momentum)
            self.running_var.mul_(1.0 - self.momentum)
            self.running_mean.add_(self.batch_mean.detach() * self.momentum)
            self.running_var.add_(self.batch_var.detach() * self.momentum)

            mean, var = self.batch_mean, self.batch_var
        else:
            mean, var = self.running_mean, self.running_var

        x = (x - mean) / torch.sqrt(var)
        x = x * torch.exp(self.log_gamma) + self.beta

        num_pixels = np.prod(x.size()) // (x.size(0) * x.size(1))
        log_det = self.log_gamma - 0.5 * torch.log(var)
        log_det_jacob += torch.sum(log_det) * num_pixels

        return x, log_det_jacob

    def backward(self, x, log_det_jacob):
        if self.training:
            mean, var = self.batch_mean, self.batch_var
        else:
            mean, var = self.running_mean, self.running_var

        x = (x - self.beta) / torch.exp(self.log_gamma)
        x = x * torch.sqrt(var) + mean

        num_pixels = np.prod(x.size()) // (x.size(0) * x.size(1))
        log_det = -self.log_gamma + 0.5 * torch.log(var)
        log_det_jacob += torch.sum(log_det) * num_pixels

        return x, log_det_jacob


class Compose(nn.Module):
    """ compose flow layers """
    def __init__(self, layers):
        super(Compose, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, z, log_df_dz):
        for layer in self.layers:
            z, log_df_dz = layer(z, log_df_dz)
        return z, log_df_dz

    def backward(self, z, log_df_dz):
        for layer in reversed(self.layers):
            z, log_df_dz = layer.backward(z, log_df_dz)
        return z, log_df_dz


class ResBlockLinear(nn.Module):
    def __init__(self, in_channels, out_channels, weight_norm=True):
        super(ResBlockLinear, self).__init__()

        self.net = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            weight_norm_wrapper(nn.Linear(in_channels, out_channels), weight_norm),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            weight_norm_wrapper(nn.Linear(out_channels, out_channels), weight_norm),
        )

        if in_channels != out_channels:
            self.bridge = weight_norm_wrapper(nn.Linear(in_channels, out_channels), weight_norm)
        else:
            self.bridge = nn.Sequential()

    def forward(self, x):
        y = self.net(x)
        x = self.bridge(x)
        return x + y


class ResBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, weight_norm=True):
        super(ResBlock2d, self).__init__()

        self.net = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            weight_norm_wrapper(nn.Conv2d(in_channels, out_channels, 3, 1, 1), weight_norm),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            weight_norm_wrapper(nn.Conv2d(out_channels, out_channels, 3, 1, 1), weight_norm),
        )

        if in_channels != out_channels:
            self.bridge = weight_norm_wrapper(nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                                              weight_norm)
        else:
            self.bridge = nn.Sequential()

    def forward(self, x):
        y = self.net(x)
        x = self.bridge(x)
        return x + y


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, base_filters=32, n_blocks=2, weight_norm=True):
        super(MLP, self).__init__()

        self.in_block = nn.Sequential(
            weight_norm_wrapper(nn.Linear(in_channels, base_filters), weight_norm),
        )

        res_blocks = []
        for _ in range(n_blocks):
            res_blocks.append(ResBlockLinear(base_filters, base_filters, weight_norm))
        self.mid_block = nn.Sequential(*res_blocks)

        self.out_block = nn.Sequential(
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True),
            weight_norm_wrapper(nn.Linear(base_filters, out_channels), weight_norm),
        )

    def forward(self, x):
        x = self.in_block(x)
        x = self.mid_block(x)
        return self.out_block(x)


class ConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_filters=32, n_blocks=2, weight_norm=True):
        super(ConvNet, self).__init__()

        self.in_block = nn.Sequential(
            weight_norm_wrapper(nn.Conv2d(in_channels, base_filters, 3, 1, 1), weight_norm),
        )

        res_blocks = []
        for i in range(n_blocks):
            res_blocks.append(ResBlock2d(base_filters, base_filters, weight_norm))
        self.mid_block = nn.Sequential(*res_blocks)

        self.out_block = nn.Sequential(
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            weight_norm_wrapper(nn.Conv2d(base_filters, out_channels, 1, 1, 0), weight_norm),
        )

    def forward(self, x):
        x = self.in_block(x)
        x = self.mid_block(x)
        return self.out_block(x)


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
            y = y.contiguous()

        num_pixels = np.prod(y.size()) // (y.size(0) * y.size(1))
        log_df_dz -= torch.sum(self.log_s, dim=0) * num_pixels

        return y, log_df_dz


class GatedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(GatedLinear, self).__init__()
        self.op = nn.Linear(in_features * 2, out_features)

    def forward(self, x):
        C = x.size(1)
        # non-linear
        y = F.elu(torch.cat([x, -x], dim=1))
        # linear
        y = self.op(y)
        # non-linear
        y = F.elu(torch.cat([y, -y], dim=1))
        # gate
        y, a = torch.split(y, C, dim=1)
        y = y * torch.sigmoid(a)
        return x + y


class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GatedConv2d, self).__init__()
        self.op = nn.Conv2d(in_channels * 2, out_channels, 3, 1, 1)

    def forward(self, x):
        C = x.size(1)
        # non-linear
        y = F.elu(torch.cat([x, -x], dim=1))
        # conv
        y = self.op(y)
        # non-linear
        y = F.elu(torch.cat([y, -y], dim=1))
        # gate
        y, a = torch.split(y, C, dim=1)
        y = y * torch.sigmoid(a)
        return x + y


class GatedAttn(nn.Module):
    def __init__(self, in_out_shape, filters=8, heads=4):
        super(GatedAttn, self).__init__()
        assert filters % heads == 0

        self.channels = in_out_shape[0]
        self.filters = filters
        self.heads = heads
        self.conv1 = nn.Conv1d(self.channels, filters * 3, 1, 1, 0)
        self.conv2 = nn.Conv1d(filters, self.channels * 2, 1, 1, 0)

        # NOTE: adding noises for positional encoding,
        # which is introduced in the following paper.
        # Gehring+ 2017, "Convolutional Sequence to Sequence Learning"
        # https://arxiv.org/abs/1705.03122
        pos_emb = nn.Parameter(torch.randn(1, *in_out_shape) * 0.01)
        self.register_parameter('pos_emb', pos_emb)

    def forward(self, x):
        org_shape = x.size()
        B = org_shape[0]
        C = org_shape[1]
        D = self.filters // self.heads
        assert C == self.channels

        # attention
        x_reshape = (x + self.pos_emb).view(B, C, -1)
        params = self.conv1(x_reshape).view(B, 3 * self.heads, D, -1)  # (B, 3head, D, *)
        V, K, Q = torch.split(params, self.heads, dim=1)  # (B, head, D, *)
        W = torch.matmul(V.permute(0, 1, 3, 2), K) / np.sqrt(D)  # (B, head, *, *)
        W = F.softmax(W, dim=2)
        A = torch.matmul(Q, W)  # (B, head, D, *)
        A = A.view(B, C, -1)

        # gate
        y = self.conv2(A)
        y, a = torch.split(y, C, dim=1)
        y = y * torch.sigmoid(a)
        y = y.view(org_shape)

        return x + y
