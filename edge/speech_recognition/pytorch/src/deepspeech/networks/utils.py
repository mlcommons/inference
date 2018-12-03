from torch import nn


def to_cuda(network):
    """Calls model.cuda() and moves input(s) to the GPU before forward."""
    network.cuda()

    network._to_cuda_forward_cache = network.forward

    def cuda_forward(x):
        return network._to_cuda_forward_cache(x.cuda(non_blocking=True))

    network.forward = cuda_forward


class OverLastDim(nn.Module):
    """Collapses a tensor to 2D, applies a module, and (re-)expands the tensor.

    An n-dimensional tensor of shape (s_1, s_2, ..., s_n) is first collapsed to
    a tensor with shape (s_1*s_2*...*s_n-1, s_n). The module is called with
    this as input producing (s_1*s_2*...*s_n-1, s_n') --- note that the final
    dimension can change. This is expanded to (s_1, s_2, ..., s_n-1, s_n') and
    returned.

    Args:
        module (nn.Module): Module to apply. Must accept a 2D tensor as input
            and produce a 2D tensor as output, optionally changing the size of
            the last dimension.
    """

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        *dims, input_size = x.size()

        reduced_dims = 1
        for dim in dims:
            reduced_dims *= dim

        x = x.view(reduced_dims, -1)
        x = self.module(x)
        x = x.view(*dims, -1)
        return x
