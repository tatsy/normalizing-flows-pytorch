import torch
import torch.nn as nn


class WeightNorm(nn.Module):
    append_g = '_g'
    append_v = '_v'

    def __init__(self, module, name='weight', dim=0, eps=1.0e-5):
        super(WeightNorm, self).__init__()
        self.module = module
        self.name = name
        self.dim = dim
        self.eps = eps
        self._reset()

    def _reset(self):
        w = getattr(self.module, self.name)

        # construct g,v such that w = g/||v|| * v
        g = torch.norm(w, dim=self.dim)
        v = w / (g.expand_as(w) + self.eps)
        g = nn.Parameter(g)
        v = nn.Parameter(v)
        name_g = self.name + self.append_g
        name_v = self.name + self.append_v

        # remove w from parameter list
        del self.module._parameters[self.name]

        # add g and v as new parameters
        self.module.register_parameter(name_g, g)
        self.module.register_parameter(name_v, v)

    def _update_weights(self):
        name_g = self.name + self.append_g
        name_v = self.name + self.append_v
        g = getattr(self.module, name_g)
        v = getattr(self.module, name_v)
        w = v * (g / (torch.norm(v, dim=self.dim) + self.eps)).expand_as(v)
        setattr(self.module, self.name, w)

    def forward(self, *args):
        self._update_weights()
        return self.module.forward(*args)
