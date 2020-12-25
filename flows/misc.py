import torch


def safe_detach(x):
    """
    detech operation which keeps reguires_grad
    ---
    https://github.com/rtqichen/residual-flows/blob/master/lib/layers/iresblock.py
    """
    return x.detach().requires_grad_(x.requires_grad)


def anomaly_hook(self, input, output):
    """
    module hook for detecting NaN and infinity
    """
    if not isinstance(output, tuple):
        outputs = [output]
    else:
        outputs = list(output)

    for i, out in enumerate(outputs):
        inf_mask = torch.isinf(out)
        nan_mask = torch.isnan(out)
        if inf_mask.any():
            print('In module:', self.__class__.__name__)
            print(f'Found NAN in output {i} at indices: ', inf_mask.nonzero(), 'where:',
                  out[inf_mask.nonzero(as_tuple=False)[:, 0].unique(sorted=True)])

        if nan_mask.any():
            print("In", self.__class__.__name__)
            print(f'Found NAN in output {i} at indices: ', nan_mask.nonzero(), 'where:',
                  out[nan_mask.nonzero(as_tuple=False)[:, 0].unique(sorted=True)])

        if inf_mask.any() or nan_mask.any():
            raise RuntimeError('Foud INF or NAN in output of', self.___class__.__name__,
                               'from input tensor', input)
