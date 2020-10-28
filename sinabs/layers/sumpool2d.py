import torch


class SumPool2d(torch.nn.LPPool2d):
    """
    Non-spiking sumpooling layer to be used in analogue Torch models. It is identical to torch.nn.LPPool2d with p=1.

    :param kernel_size: the size of the window
    :param stride: the stride of the window. Default value is kernel_size
    :param ceil_mode: when True, will use ceil instead of floor to compute the output shape
    """
    def __init__(self, kernel_size, stride=None, ceil_mode=False):
        super().__init__(1, kernel_size, stride, ceil_mode)
