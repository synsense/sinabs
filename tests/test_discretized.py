from sinabs.backend.speck import discretize
import numpy as np
import torch
from sinabs.layers import SpikingLayerBPTT

# - Test tensor to be discretized
float_tensor = torch.tensor(
    [
        [
            [13.5594, -39.0465, -31.1756, 41.0779],
            [34.7378, -14.0715, -13.4176, 26.0796],
            [-17.0537, 2.0017, 2.1757, 33.9470],
        ],
        [
            [-13.5850, 36.3960, 29.9012, -8.1596],
            [-19.3947, -15.8194, -8.1501, -2.0120],
            [8.2892, -13.3248, 18.3985, -11.8316],
        ],
    ]
)

## -- Layers to be discretized

torch.manual_seed(8)
# - Convolutional
weight = torch.randn(4, 2, 3, 3) * 10
bias = torch.randn(4) * 10
conv_lyr = torch.nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(3, 3))
conv_lyr.weight = torch.nn.Parameter(weight)
conv_lyr.bias = torch.nn.Parameter(bias)

# - Spiking layer
thr = np.random.random() * 10
thr_low = -np.random.random() * 10
spk_lyr = SpikingLayerBPTT(threshold=thr, threshold_low=thr_low, membrane_subtract=True)

# - Evolve to initialize layer state
inp = torch.randn(30, 2, 32, 32)
out_conv = conv_lyr(inp)
out_spk = spk_lyr(inp)


def validate_common_scaling(conv_lyr, spk_lyr, weight, bias, thr, thr_low, state=None):
    weight_scalings = weight / conv_lyr.weight
    if bias is not None:
        bias_scaling = bias / conv_lyr.bias
    thr_scaling = thr / spk_lyr.threshold
    thr_low_scaling = thr_low / spk_lyr.threshold_low


def test_discretize_conv_spike():
    # - Discretization of convolutional and spiking layer
    conv_discr, spk_discr = discretize.discretize_conv_spike(conv_lyr, spk_lyr)


def test_discr_tensor():
    scaling = 3.0917

    discr_tensor = discretize.discretize_tensor(float_tensor, scaling, to_int=False)

    assert torch.isclose(discr_tensor, float_tensor * scaling)
    assert discr_tensor.dtype == torch.float

    discr_tensor_int = discretize.discretize_tensor(float_tensor, scaling, to_int=True)

    assert torch.isclose(discr_tensor_int, float_tensor * scaling)
    assert discr_tensor_int.dtype == torch.int


def test_discr_scalar():
    scaling = 3.0917

    scalar = 13.5594
    scal_disc = discretize.discretize_scalar(scalar, scaling)
    assert scal_disc == 42

    scalar_neg = -13.5594
    scal_disc_neg = discretize.discretize_scalar(scalar_neg, scaling)
    assert scal_disc_neg == -42


def test_scaling():
    bit_precision = 8

    def test_obj(obj):
        scaling = discretize.determine_discretization_scale(obj, bit_precision)
        obj_scaled = scaling * obj

        assert (obj_scaled >= -128).all()
        assert (obj_scaled <= 127).all()
        assert torch.isclose(
            torch.min(obj_scaled), torch.tensor(-128.0)
        ) or torch.isclose(torch.max(obj_scaled), torch.tensor(127.0))

    test_obj(float_tensor)
    test_obj(float_tensor + 20)
    test_obj(float_tensor - 20)
    test_obj(float_tensor / 20)
    test_obj(float_tensor * 20)
