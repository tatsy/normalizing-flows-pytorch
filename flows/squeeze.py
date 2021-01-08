import torch
import torch.nn as nn


def channel_split(z, dim=1, odd=False):
    C = z.size(dim)
    z0, z1 = torch.split(z, C // 2, dim=dim)
    if odd:
        z0, z1 = z1, z0
    return z0, z1


def channel_merge(z0, z1, dim=1, odd=False):
    if odd:
        z0, z1 = z1, z0
    z = torch.cat([z0, z1], dim=dim)
    return z


def get_checker_mask(H, W, odd=False, device=None):
    ix = torch.arange(W).to(device).long()
    iy = torch.arange(H).to(device).long()
    iy, ix = torch.meshgrid([iy, ix])

    mod = 0 if odd else 1
    mask = ((ix + iy) % 2 == mod).float()
    mask = mask.view(1, 1, H, W)

    return mask


def checker_split(z, odd=False):
    assert z.dim() == 4
    B, C, H, W = z.size()

    z = z.view(B, C, H // 2, 2, W // 2, 2)  # (B, C, sH, 2, sW, 2)
    z = z.permute(0, 1, 3, 5, 2, 4).contiguous()  # (B, C, 2, 2, sH, sW)
    z = z.view(B, C * 4, H // 2, W // 2)  # (B, C * 4, sH, sW)
    za, zb, zc, zd = torch.split(z, C, dim=1)
    z0 = torch.cat([za, zd], dim=1)
    z1 = torch.cat([zb, zc], dim=1)
    if odd:
        z0, z1 = z1, z0
    return z0, z1


def checker_merge(z0, z1, odd=False):
    assert z0.dim() == 4 and z1.dim() == 4
    B, C2, sH, sW = z0.size()
    C = C2 // 2

    if odd:
        z0, z1 = z1, z0

    za, zd = torch.split(z0, C, dim=1)
    zb, zc = torch.split(z1, C, dim=1)
    z = torch.cat([za, zb, zc, zd], dim=1)

    z = z.view(B, C, 2, 2, sH, sW).permute(0, 1, 4, 2, 5, 3).contiguous()
    z = z.view(B, C, sH * 2, sW * 2)
    return z


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
    z = z.view(B, -1).contiguous()
    return z


def squeeze2d(z, odd=False):
    assert z.dim() == 4
    B, C, H, W = z.size()

    z = z.view(B, C, H // 2, 2, W // 2, 2)  # (B, C, sH, 2, sW, 2)
    z = z.permute(0, 1, 3, 5, 2, 4).contiguous()  # (B, C, 2, 2, sH, sW)
    z = z.view(B, C * 4, H // 2, W // 2)  # (B, C * 4, sH, sW)
    z0, z1 = torch.split(z, C * 2, dim=1)
    if odd:
        z0, z1 = z1, z0
    return z0, z1


def unsqueeze2d(z0, z1, odd=False):
    assert z0.dim() == 4 and z1.dim() == 4
    B, C2, sH, sW = z0.size()
    C = C2 // 2

    if odd:
        z0, z1 = z1, z0

    z = torch.cat([z0, z1], dim=1)

    z = z.view(B, C, 2, 2, sH, sW).permute(0, 1, 4, 2, 5, 3).contiguous()
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
