# author    : Willian Soares Girao
# contact   : williansoaresgirao@gmail.com

import statistics
from typing import Iterable

import numpy as np


def rescale_method_1(scaling_factors: Iterable[int], lambda_: float = 0.5) -> float:
    """
        This method will use the average (scaled by `lambda_`) of the computed re-scaling factor
    for the pooling layer(s) feeding into a convolutional layer.

    Arguments
    ---------
    - scaling_factors (list): the list of re-scaling factors computed by each `SumPool2d` layer targeting a
        single `Conv2d` layer within a `DynapcnnLayer` instance.
    - lambda_ (float): a scaling variable that multiplies the computed average re-scaling factor of the pooling layers.

    Returns
    ---------
    - the averaged re-scaling factor multiplied by `lambda_` if `len(scaling_factors) > 0`, else `1` is returned.
    """

    if len(scaling_factors) > 0:
        return np.round(np.mean(list(scaling_factors)) * lambda_, 2)
    else:
        return 1.0


def rescale_method_2(scaling_factors: Iterable[int], lambda_: float = 0.5) -> float:
    """
        This method will use the harmonic mean (scaled by `lambda_`) of the computed re-scaling factor
    for the pooling layer(s) feeding into a convolutional layer.

    Arguments
    ---------
    - scaling_factors (list): the list of re-scaling factors computed by each `SumPool2d` layer targeting a
        single `Conv2d` layer within a `DynapcnnLayer` instance.
    - lambda_ (float): a scaling variable that multiplies the computed average re-scaling factor of the pooling layers.

    Returns
    ---------
    - the averaged re-scaling factor multiplied by `lambda_` if `len(scaling_factors) > 0`, else `1` is returned.

    Note
    ---------
    - since the harmonic mean is less sensitive to outliers it **could be** that this is a better method
        for weight re-scaling when multiple poolings with big differentces in kernel sizes are being considered.
    """

    if len(scaling_factors) > 0:
        return np.round(statistics.harmonic_mean(list(scaling_factors)) * lambda_, 2)
    else:
        return 1.0
