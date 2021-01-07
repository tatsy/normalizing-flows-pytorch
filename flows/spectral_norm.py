import torch
import torch.nn as nn


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    """
    modified spectral normalization [Miyato et al. 2018] for invertible residual networks
    ---    
    most of this implementation is borrowed from the following link:
    https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    """
    def __init__(self, module, coeff=0.97, eps=1.0e-5, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.coeff = coeff
        self.eps = eps
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + '_u')
        v = getattr(self.module, self.name + '_v')
        w = getattr(self.module, self.name + '_bar')

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        scale = self.coeff / (sigma + self.eps)

        delattr(self.module, self.name)
        if scale < 1.0:
            setattr(self.module, self.name, w * scale.expand_as(w))
        else:
            setattr(self.module, self.name, w)

    def _made_params(self):
        try:
            _ = getattr(self.module, self.name + '_u')
            _ = getattr(self.module, self.name + '_v')
            _ = getattr(self.module, self.name + '_bar')
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = w.data.new(height).normal_(0, 1)
        v = w.data.new(width).normal_(0, 1)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        self.module.register_buffer(self.name + '_u', u)
        self.module.register_buffer(self.name + '_v', v)
        self.module.register_parameter(self.name + '_bar', w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)
