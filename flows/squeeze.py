import torch
import torch.nn as nn


def squeeze1d(z, odd=False):
    assert z.dim() == 2
    B, C = z.size()
    z = z.view(B, C // 2, 2)
    z0 = z[:, :, 0]
    z1 = z[:, :, 1]
    if odd:
        z0, z1 = z1, z0
    return z0, z1


def unsqueeze1d(z0, z1, odd=False):
    assert z0.dim() == 2 and z1.dim() == 2
    B, hC = z0.size()

    if odd:
        z0, z1 = z1, z0
    z = torch.stack([z0, z1], dim=-1)
    z = z.view(B, -1)
    return z


def squeeze2d(z, odd=False):
    assert z.dim() == 4
    B, C, H, W = z.size()

    z = z.unfold(2, 2, 2).unfold(3, 2, 2).contiguous()  # (B, C, sH, sW, 2, 2)
    z = z.view(B, C, H // 2, W // 2, 4)
    za, zb, zc, zd = torch.split(z, 1, dim=4)
    z0 = torch.cat([za, zd], dim=1).squeeze(-1)
    z1 = torch.cat([zb, zc], dim=1).squeeze(-1)
    if odd:
        z0, z1 = z1, z0
    return z0, z1


def unsqueeze2d(z0, z1, odd=False):
    assert z0.dim() == 4 and z1.dim() == 4
    B, C2, sH, sW = z0.size()
    C = C2 // 2

    if odd:
        z0, z1 = z1, z0
    za, zd = torch.split(z0, C, dim=1)
    zb, zc = torch.split(z1, C, dim=1)
    z = torch.stack([za, zb, zc, zd], dim=-1)  # (B, C, sH, sW, 4)
    z = z.view(B, C, sH, sW, 2, 2).permute(0, 1, 2, 4, 3, 5).contiguous()
    z = z.view(B, C, sH * 2, sW * 2)
    return z


class Squeeze1d(nn.Module):
    """
    split 1D vector into two half-size vectors
    by extracting entries alternatingly
    """
    def __init__(self, odd=False):
        super(Squeeze1d, self).__init__()
        self.odd = odd

    def forward(self, z, log_df_dz):
        z0, z1 = squeeze1d(z, self.odd)
        z = torch.cat([z0, z1], dim=1)
        return z, log_df_dz

    def backward(self, z, log_df_dz):
        z0, z1 = torch.split(z, z.size(1) // 2, dim=1)
        z = unsqueeze1d(z0, z1, self.odd)
        return z, log_df_dz


class Unsqueeze1d(nn.Module):
    """
    merge 1D vectors given by Squeeze1d
    """
    def __init__(self, odd=False):
        super(Unsqueeze1d, self).__init__()
        self.odd = odd

    def forward(self, z, log_df_dz):
        z0, z1 = torch.split(z, z.size(1) // 2, dim=1)
        z = unsqueeze1d(z0, z1, self.odd)
        return z, log_df_dz

    def backward(self, z, log_df_dz):
        z0, z1 = squeeze1d(z, self.odd)
        z = torch.cat([z0, z1], dim=1)
        return z, log_df_dz


class Squeeze2d(nn.Module):
    """
    split an 2D feature map into two maps by
    extracting pixels using checkerboard pattern
    """
    def __init__(self, odd=False):
        super(Squeeze2d, self).__init__()
        self.odd = odd

    def forward(self, z, log_df_dz):
        z0, z1 = squeeze2d(z, self.odd)
        z = torch.cat([z0, z1], dim=1)
        return z, log_df_dz

    def backward(self, z, log_df_dz):
        z0, z1 = torch.split(z, z.size(1) // 2, dim=1)
        z = unsqueeze2d(z0, z1, self.odd)
        return z, log_df_dz


class Unsqueeze2d(nn.Module):
    """
    Merge two 2D feature maps given by Squeeze2d
    """
    def __init__(self, odd=False):
        super(Unsqueeze2d, self).__init__()
        self.odd = odd

    def forward(self, z, log_df_dz):
        z0, z1 = torch.split(z, z.size(1) // 2, dim=1)
        z = unsqueeze2d(z0, z1, self.odd)
        return z, log_df_dz

    def backward(self, z, log_df_dz):
        z0, z1 = squeeze2d(z, self.odd)
        z = torch.cat([z0, z1], dim=1)
        return z, log_df_dz
