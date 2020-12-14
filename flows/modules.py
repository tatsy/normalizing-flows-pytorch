import torch
import torch.nn as nn


def deriv_sigmoid(x, eps=1.0e-6):
    sigma = torch.sigmoid(x)
    return torch.clamp(sigma * (1.0 - sigma), min=eps, max=1.0 - eps)


def deriv_logit(x, eps=1.0e-6):
    y = torch.logit(torch.clamp(x, 0.0 + eps, 1.0 - eps))
    return 1.0 / deriv_sigmoid(y, eps)
    


class WeightNormLinear(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 bias=True,
                 weight_norm=True,
                 scale=False):
        """Intializes a Conv1d augmented with weight normalization.
        (See torch.nn.utils.weight_norm for detail.)
        Args:
            in_dim: number of input channels.
            out_dim: number of output channels.
            bias: True if include learnable bias parameters, False otherwise.
            weight_norm: True if apply weight normalization, False otherwise.
            scale: True if include magnitude parameters, False otherwise.
        """
        super(WeightNormLinear, self).__init__()

        if weight_norm:
            self.conv = nn.utils.weight_norm(
                nn.Linear(in_dim, out_dim, bias=bias))
            if not scale:
                self.conv.weight_g.data = torch.ones_like(
                    self.conv.weight_g.data)
                self.conv.weight_g.requires_grad = False  # freeze scaling
        else:
            self.conv = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x):
        """Forward pass.
        Args:
            x: input tensor.
        Returns:
            transformed tensor.
        """
        return self.conv(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()

        self.net = nn.Sequential(nn.BatchNorm1d(in_channels),
                                 nn.ReLU(inplace=True),
                                 WeightNormLinear(in_channels, out_channels),
                                 nn.BatchNorm1d(out_channels),
                                 nn.ReLU(inplace=True),
                                 WeightNormLinear(out_channels, out_channels))

        if in_channels != out_channels:
            self.bridge = nn.Sequential(
                WeightNormLinear(in_channels, out_channels), )
        else:
            self.bridge = nn.Sequential()

    def forward(self, x):
        y = self.net(x)
        x = self.bridge(x)
        return x + y


class Network(nn.Module):
    def __init__(self, in_channels, out_channels, base_filters=32):
        super(Network, self).__init__()

        self.in_block = nn.Sequential(
            WeightNormLinear(in_channels, base_filters),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True),
            WeightNormLinear(base_filters, base_filters),
        )

        self.mid_block = nn.Sequential(ResBlock(base_filters, base_filters),
                                       ResBlock(base_filters, base_filters))

        self.out_block = nn.Sequential(
            nn.BatchNorm1d(base_filters), nn.ReLU(inplace=True),
            WeightNormLinear(base_filters, out_channels))

    def forward(self, x):
        x = self.in_block(x)
        x = self.mid_block(x)
        return self.out_block(x)
