import numpy as np
import statistics

def rescale_method_1(rescaling_from_sumpool: list, lambda_: float = 0.5) -> float:
    """
        The `method 1` will use the average of the computed rescaling factor for each pooling layer
    feeding into a convolutional layer (if there are more than one)...

    Arguments
    ---------
    """

    if len(rescaling_from_sumpool):
        return np.round(np.mean(rescaling_from_sumpool)*lambda_, 2)
    else:
        return 1

def rescale_method_2(rescaling_from_sumpool: list, lambda_: float = 0.5) -> float:
    """
        The `method 2` will use the harmonic mean of the computed rescaling factor for each pooling layer
    feeding into `conv_layer` (if there are more than one) ...

    Note: since the harmonic mean is less sensitive to outliers it **could be** that this is a better method
    for weight re-scaling when multiple pooling with big differentces in kernel sizes are being considered.

    Arguments
    ---------
    """

    if len(rescaling_from_sumpool):
        return np.round(statistics.harmonic_mean(rescaling_from_sumpool)*lambda_, 2)
    else:
        return 1