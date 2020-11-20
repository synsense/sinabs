from copy import deepcopy
import numpy as np
import torch
from sinabs.backend.dynapcnn import discretize
from sinabs.layers import SpikingLayer

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

MIN_STATE = -(2 ** (discretize.DYNAPCNN_STATE_PRECISION_BITS - 1))
MAX_STATE = -MIN_STATE - 1
MIN_WEIGHT = -(2 ** (discretize.DYNAPCNN_WEIGHT_PRECISION_BITS - 1))
MAX_WEIGHT = -MIN_WEIGHT - 1

# -- Layers to be discretized
torch.manual_seed(8)
# - Convolutional
weight = torch.randn(4, 2, 3, 3) * 10
bias = torch.randn(4) * 10
conv_lyr = torch.nn.Conv2d(
    in_channels=2, out_channels=4, kernel_size=(3, 3), bias=False
)
conv_lyr.weight = torch.nn.Parameter(weight)

# - Spiking layer
np.random.seed(4)
thr = np.random.random() * 10
thr_low = -np.random.random() * 10
spk_lyr = SpikingLayer(threshold=thr, threshold_low=thr_low, membrane_subtract=True)


def validate_common_scaling(conv_lyr, spk_lyr, weight, bias, thr, thr_low, state):
    """
    Make sure that all discretized values are scaled by same factor and within
    their allowed range.
    """

    if thr is not None:
        # Guess scaling based on threshold
        scale = thr / spk_lyr.threshold
        reference = spk_lyr.threshold

        thrs = torch.tensor((thr, thr_low))
        thrs_old = torch.tensor((spk_lyr.threshold, spk_lyr.threshold_low))
    else:
        with torch.no_grad():
            scale = torch.true_divide(torch.max(weight), torch.max(conv_lyr.weight))
            reference = torch.max(weight)
            thrs = thrs_old = None

    # Error should be within what is expected from rounding
    for orig, new in zip(
        (conv_lyr.weight, conv_lyr.bias, spk_lyr.state, thrs_old),
        (weight, bias, state, thrs),
    ):
        if new is not None:
            # Explanation for the tolerance:
            # Let ref_d be the scaled and discretized `reference` value, and `s` the (true)
            # scaling factor. Then the difference `diff` between ref_d and reference * scaling
            # is due to rounding and at most 0.5. Therefore the guessed scaling factor
            # `scale` is ref_d / reference = (ref_d + diff) / reference
            # When multiplying it with another original value it will deviate at most by
            # orig * |scale - s| + 0.5, where 0.5 is from rounding and scale - s the
            # difference between guessed and true scaling factor. |scale - s| = |diff| / |reference|
            # and |diff| <= 0.5, therefore the total error is at most 0.5 * |orig/referece| + 0.5
            tol = 0.5 * (1 + torch.abs(torch.true_divide(new, reference)))
            assert (torch.abs(scale * orig - new) <= tol).all()
    # Make sure that new values are in allowed range
    for new in (weight, bias):
        if new is not None:
            assert (new <= MAX_WEIGHT).all()
            assert (new >= MIN_WEIGHT).all()
    for new in (thrs, state):
        if new is not None:
            assert (state <= MAX_STATE).all()
            assert (state >= MIN_STATE).all()


def validate_discretization(conv_lyr, spk_lyr, inplace=False, to_int=True):
    """Discretize layers and make sure the discretized values are sensible."""
    if inplace:
        conv_discr, spk_discr = discretize.discretize_conv_spike_(
            conv_lyr, spk_lyr, to_int=to_int
        )
    else:
        conv_discr, spk_discr = discretize.discretize_conv_spike(
            conv_lyr, spk_lyr, to_int=to_int
        )
    validate_common_scaling(
        conv_lyr,
        spk_lyr,
        conv_discr.weight,
        conv_discr.bias,
        spk_discr.threshold,
        spk_discr.threshold_low,
        spk_discr.state,
    )
    return conv_discr, spk_discr


def test_discretize_conv_spike():
    """Test joint discretization of spiking and convolutional layers"""

    # - Make copies of original layers
    conv_copy = deepcopy(conv_lyr)
    spk_copy = deepcopy(spk_lyr)

    # - Discretization of convolutional and spiking layer
    validate_discretization(conv_copy, spk_copy)

    # - Set bias
    conv_lyr.bias = torch.nn.Parameter(bias)
    conv_copy.bias = torch.nn.Parameter(bias)
    validate_discretization(conv_copy, spk_copy)

    # - Evolve to initialize layer state
    inp = torch.randn(30, 2, 32, 32)
    out_conv = conv_copy(inp)
    spk_copy(out_conv)
    spk_lyr(out_conv)
    conv_discr, spk_discr = validate_discretization(conv_copy, spk_copy)

    # Make sure that discretization did not happen in-place
    assert (conv_lyr.weight == conv_copy.weight).all()
    assert (conv_lyr.bias == conv_copy.bias).all()
    assert (spk_lyr.state == spk_copy.state).all()
    assert spk_lyr.threshold == spk_copy.threshold
    assert spk_lyr.threshold_low == spk_copy.threshold_low

    # Make sure that elements are integers
    for obj in (conv_discr.weight, conv_discr.bias, spk_discr.state):
        assert obj.dtype == torch.int
    for obj in (spk_discr.threshold, spk_discr.threshold):
        assert isinstance(obj, (int, np.integer))

    # - In-place mutations
    conv_discr, spk_discr = validate_discretization(conv_copy, spk_copy, inplace=True)

    # Make sure that discretization did happen in-place
    assert (conv_discr.weight == conv_copy.weight).all()
    assert (conv_discr.bias == conv_copy.bias).all()
    assert (spk_discr.state == spk_copy.state).all()
    assert spk_discr.threshold == spk_copy.threshold
    assert spk_discr.threshold_low == spk_copy.threshold_low

    # - No conversion to integers
    conv_copy = deepcopy(conv_lyr)
    spk_copy = deepcopy(spk_lyr)

    conv_discr, spk_discr = validate_discretization(conv_copy, spk_copy, to_int=False)

    # Make sure that discretization did not happen in-place
    assert (conv_lyr.weight == conv_copy.weight).all()
    assert (conv_lyr.bias == conv_copy.bias).all()
    assert (spk_lyr.state == spk_copy.state).all()
    assert spk_lyr.threshold == spk_copy.threshold
    assert spk_lyr.threshold_low == spk_copy.threshold_low

    # Make sure that elements are floats
    for obj in (conv_discr.weight, conv_discr.bias, spk_discr.state):
        assert obj.dtype == torch.float


def test_discr_conv():
    """Discretization of only convolutional layer"""

    # - Add biases and evolve to initialize layer state
    conv_lyr.bias = torch.nn.Parameter(bias)

    conv_copy = deepcopy(conv_lyr)
    spk_copy = deepcopy(spk_lyr)

    inp = torch.randn(30, 2, 32, 32)
    out_conv = conv_copy(inp)
    spk_copy(out_conv)
    spk_lyr(out_conv)

    conv_discr = discretize.discretize_conv(
        conv_copy, spk_copy.threshold, spk_copy.threshold_low, spk_copy.state
    )
    validate_common_scaling(
        conv_copy, spk_copy, conv_discr.weight, conv_discr.bias, None, None, None,
    )

    # Make sure that discretization did not happen in-place
    assert (conv_lyr.weight == conv_copy.weight).all()
    assert (conv_lyr.bias == conv_copy.bias).all()

    # - In-place
    conv_discr = discretize.discretize_conv_(
        conv_copy, spk_copy.threshold, spk_copy.threshold_low, spk_copy.state
    )
    validate_common_scaling(
        conv_copy, spk_copy, conv_discr.weight, conv_discr.bias, None, None, None,
    )

    # Make sure that discretization did happen in-place
    assert (conv_discr.weight == conv_copy.weight).all()
    assert (conv_discr.bias == conv_copy.bias).all()

    # - Make sure that spike layer elements did not get mutated
    assert (spk_lyr.state == spk_copy.state).all()
    assert spk_lyr.threshold == spk_copy.threshold
    assert spk_lyr.threshold_low == spk_copy.threshold_low


def test_discr_spk():
    """Discretization of only spiking layer"""

    # - Add biases and evolve to initialize layer state
    conv_lyr.bias = torch.nn.Parameter(bias)

    conv_copy = deepcopy(conv_lyr)
    spk_copy = deepcopy(spk_lyr)

    inp = torch.randn(30, 2, 32, 32)
    out_conv = conv_copy(inp)
    spk_copy(out_conv)
    spk_lyr(out_conv)

    conv_copy = deepcopy(conv_lyr)
    spk_copy = deepcopy(spk_lyr)

    spk_discr = discretize.discretize_spk(spk_copy, conv_copy.weight, conv_copy.bias)
    validate_common_scaling(
        conv_copy,
        spk_copy,
        None,
        None,
        spk_discr.threshold,
        spk_discr.threshold_low,
        spk_discr.state,
    )

    # Make sure that discretization did not happen in-place
    assert (spk_lyr.state == spk_copy.state).all()
    assert spk_lyr.threshold == spk_copy.threshold
    assert spk_lyr.threshold_low == spk_copy.threshold_low

    # - In-place
    spk_discr = discretize.discretize_spk_(spk_copy, conv_copy.weight, conv_copy.bias)
    validate_common_scaling(
        conv_copy,
        spk_copy,
        None,
        None,
        spk_discr.threshold,
        spk_discr.threshold_low,
        spk_discr.state,
    )

    # Make sure that discretization did happen in-place
    assert (spk_discr.state == spk_copy.state).all()
    assert spk_discr.threshold == spk_copy.threshold
    assert spk_discr.threshold_low == spk_copy.threshold_low

    # - Make sure that conv layer elements did not get mutated
    assert (conv_lyr.weight == conv_copy.weight).all()
    assert (conv_lyr.bias == conv_copy.bias).all()


def test_discr_tensor():
    """Discretization of a tensor object by given scaling factor"""

    scaling = 3.0917

    discr_tensor = discretize.discretize_tensor(float_tensor, scaling, to_int=False)

    assert torch.isclose(discr_tensor, float_tensor * scaling, atol=0.5).all()
    assert discr_tensor.dtype == torch.float

    discr_tensor_int = discretize.discretize_tensor(float_tensor, scaling, to_int=True)

    assert torch.isclose(
        discr_tensor_int.float(), float_tensor * scaling, atol=0.5
    ).all()
    assert discr_tensor_int.dtype == torch.int


def test_discr_scalar():
    """Discretization of a scalar object by given scaling factor"""

    scaling = 3.0917

    scalar = 13.5594
    scal_disc = discretize.discretize_scalar(scalar, scaling)
    assert scal_disc == 42

    scalar_neg = -13.5594
    scal_disc_neg = discretize.discretize_scalar(scalar_neg, scaling)
    assert scal_disc_neg == -42


def test_scaling():
    """Choice of scaling factor"""

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
