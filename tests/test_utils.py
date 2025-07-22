import torch
import torch.nn as nn

import sinabs
import sinabs.layers as sl
import sinabs.utils as utils


class SNN(nn.Module):
    def __init__(self, hidden_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            sl.IAF(),
            nn.Linear(hidden_dim, hidden_dim),
            sl.IAF(),
            nn.Linear(hidden_dim, hidden_dim),
            sl.IAF(),
            nn.Linear(hidden_dim, 5),
        )
        self.spike_output = sl.IAF()

    def forward(self, x):
        return self.spike_output(self.net(x))


def test_reset_states():
    model = SNN()
    assert model.net[1].v_mem.sum() == 0

    input = torch.rand((5, 10, 2))
    model(input)
    assert model.net[1].v_mem.grad_fn is not None
    assert model.net[1].v_mem.sum() != 0

    sinabs.reset_states(model)
    assert model.net[1].v_mem.grad_fn is None
    assert model.net[1].v_mem.sum() == 0


def test_zero_grad():
    model = SNN()
    assert model.net[1].v_mem.grad == None

    input = torch.rand((5, 10, 2))
    output = model(input)

    assert model.net[1].v_mem.grad_fn is not None
    assert model.net[1].v_mem.sum() != 0

    sinabs.zero_grad(model)
    assert model.net[1].v_mem.grad_fn is None
    assert model.net[1].v_mem.sum() != 0


def test_validate_memory_speck_raise_exception():
    import pytest

    kernel_size = [3, 3]
    stride = [1, 1]
    padding = [1, 1]
    input_feature_size = 64
    output_feature_size = 65
    input_dimension = [8, 8]

    with pytest.raises(Exception) as info:
        utils.validate_memory_mapping_speck(
            input_feature_size,
            output_feature_size,
            kernel_size,
            stride,
            padding,
            input_dimension,
        )
    assert (
        str(info.value)
        == "Kernel memory is 128Ki and can not be mapped on chip. Kernel memory on chip needs to be at most 64Ki."
    )


def test_validate_memory_speck_no_exception():
    kernel_size = [3, 3]
    stride = [1, 1]
    padding = [1, 1]
    input_feature_size = 2
    output_feature_size = 64
    input_dimension = [8, 8]

    msg = utils.validate_memory_mapping_speck(
        input_feature_size,
        output_feature_size,
        kernel_size,
        stride,
        padding,
        input_dimension,
    )

    assert (
        msg
        == "Layer can be mapped successfully. Kernel memory is 2Ki and neuron memory is 4Ki."
    )
