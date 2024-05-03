import torch.nn as nn
import numpy as np
import statistics

def rescale_method_1(conv_layer: nn.Conv2d, input_pool_kernel: list, lambda_: float = 1):
    """
        The `method 1` will use the average of the computed rescaling factor for each pooling layer
    feeding into `conv_layer` (if there are more than one) to rescale its weights.

    Arguments
    ---------
        input_pool_kernel (list): the kernels of all pooling layers feeding input to `conv_layer`.
        lambda_ (float): scales the computed re-scaling factor. If the outputs of the pooling are too small
            the rescaling might lead to vanishing gradients, so we can try to control that by scaling it by
            lambda.
    """
    rescaling_factors = []

    for kernel in input_pool_kernel:
        rescaling_factors.append(kernel[0]*kernel[1])

    rescaling_factor = np.mean(rescaling_factors)*lambda_

    # print(f'method 1 - recaling factor: {rescaling_factor} (computed using {len(input_pool_kernel)} kernels and lambda {lambda_})')

    conv_layer.weight.data /= rescaling_factor

def rescale_method_2(conv_layer: nn.Conv2d, input_pool_kernel: list, lambda_: float = 1):
    """
        The `method 2` will use the harmonic mean of the computed rescaling factor for each pooling layer
    feeding into `conv_layer` (if there are more than one) to rescale its weights.

    Arguments
    ---------
        input_pool_kernel (list): the kernels of all pooling layers feeding input to `conv_layer`.
        lambda_ (float): scales the computed re-scaling factor. If the outputs of the pooling are too small
            the rescaling might lead to vanishing gradients, so we can try to control that by scaling it by
            lambda.
    """
    rescaling_factors = []

    for kernel in input_pool_kernel:
        rescaling_factors.append(kernel[0]*kernel[1])

    rescaling_factor = statistics.harmonic_mean(rescaling_factors)*lambda_

    # print(f'method 2 - recaling factor: {rescaling_factor} (computed using {len(input_pool_kernel)} kernels and lambda {lambda_})')

    conv_layer.weight.data /= rescaling_factor