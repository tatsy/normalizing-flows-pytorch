import torch
import torch.nn as nn


class Squeeze1d(nn.Module):
    def __init__(self, dims, odd=False):
        super(Squeeze1d, self).__init__()

        assert len(dims) == 1, 'Squeeze1 can be applied to 1D tensor!'
        self.dims = dims
        self.modulo = 0 if not odd else 1

        d = dims[0]
        idx = torch.arange(d, dtype=torch.long)
        mask = torch.where(idx % 2 == self.modulo, torch.ones(d), torch.zeros(d)).long()
        self.register_buffer('mask', mask)

    def forward(self, x):
        raise NotImplementedError()

    def split(self, z):
        z0 = z[:, self.mask != 0]
        z1 = z[:, self.mask == 0]
        return z0, z1

    def merge(self, z0, z1):
        batch_size = z0.size(0)
        z = torch.zeros((batch_size, self.dims[0]), dtype=torch.float32)
        z = z.to(z0.device)
        z[:, self.mask != 0] = z0
        z[:, self.mask == 0] = z1
        return z


class Squeeze2d(nn.Module):
    def __init__(self, dims, odd=False):
        super(Squeeze2d, self).__init__()

        assert len(dims) == 3, 'Squeeze1 can be applied to 3D tensor!'
        modulo = 0 if not odd else 1
        _, h, w = dims
        self.dims = dims
        idx = torch.arange(4, dtype=torch.long)
        mask = torch.where(idx % 2 == modulo, torch.ones(4), torch.zeros(4)).long()
        self.register_buffer('mask', mask)

    def forward(self, x):
        raise NotImplementedError()

    def split(self, z):
        z_patches = z.unfold(2, 2, 2).unfold(3, 2, 2).contiguous()
        B, C, sH, sW, _, _ = z_patches.size()
        z_patches = z_patches.view(B, C, sH, sW, 4).permute(0, 1, 4, 2, 3)
        z0 = z_patches[:, :, self.mask != 0, :, :].view(B, -1, sH, sW)
        z1 = z_patches[:, :, self.mask == 0, :, :].view(B, -1, sH, sW)
        return z0, z1

    def merge(self, z0, z1):
        batch_size = z0.size(0)
        z = torch.zeros((batch_size, *self.dims), dtype=torch.float32)
        z = z.to(z0.device)

        z_patches = z.unfold(2, 2, 2).unfold(3, 2, 2).contiguous()
        B, C, sH, sW, _, _ = z_patches.size()
        z_patches = z_patches.view(B, C, sH, sW, 4).permute(0, 1, 4, 2, 3)
        z_patches[:, :, self.mask != 0, :, :] = z0.view(B, C, -1, sH, sW)
        z_patches[:, :, self.mask == 0, :, :] = z1.view(B, C, -1, sH, sW)
        z = z_patches.view(B, C, 2, 2, sH, sW).permute(0, 1, 4, 2, 5, 3)
        z = z.contiguous()
        z = z.view(B, *self.dims)

        return z
