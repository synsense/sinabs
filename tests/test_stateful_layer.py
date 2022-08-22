import torch
import torch.nn as nn
from sinabs.layers import StatefulLayer, LIF
import pytest
import numpy as np


def test_stateful_layer():
    layer = StatefulLayer(state_names=["v_mem"])
    assert not layer.is_state_initialised()
    assert layer.v_mem.shape == torch.zeros((0)).shape

    layer.reset_states()
    assert not layer.is_state_initialised()
    assert layer.v_mem.shape == torch.zeros((0)).shape


def test_lif_reset():
    layer = LIF(tau_mem=30.0)
    assert not layer.is_state_initialised()
    assert layer.v_mem.shape == torch.zeros((0)).shape

    layer(torch.rand((2, 3, 4)))
    assert layer.is_state_initialised()
    assert layer.v_mem.shape == (2, 4)
    assert layer.v_mem.sum() != 0

    layer.reset_states()
    assert layer.is_state_initialised()
    assert layer.v_mem.shape == (2, 4)
    assert layer.v_mem.sum() == 0

    layer.reset_states(randomize=True)
    assert layer.is_state_initialised()
    assert layer.v_mem.shape == (2, 4)
    assert layer.v_mem.sum() != 0


def test_changing_batch_size():
    layer = LIF(tau_mem=30.0)
    assert not layer.is_state_initialised()
    assert layer.v_mem.shape == torch.zeros((0)).shape

    layer(torch.rand((2, 3, 4)))
    assert layer.is_state_initialised()
    assert layer.v_mem.shape == (2, 4)

    layer(torch.rand((5, 3, 4)))
    assert layer.is_state_initialised()
    assert layer.v_mem.shape == (5, 4)


def test_reset_states():
    layer = StatefulLayer(state_names=["v_mem", "i_syn"])
    layer.init_state_with_shape((1, 2, 3))
    assert (1, 2, 3) == layer.v_mem.shape
    assert (1, 2, 3) == layer.i_syn.shape
    # Reset states
    layer.reset_states(randomize=False)
    assert layer.v_mem.any() == False
    # Reset states
    layer.reset_states(randomize=True)
    assert layer.v_mem.any() == True

    # Reset states
    layer.reset_states(randomize=True, value_ranges={"v_mem": (-5, -3)})
    assert layer.v_mem.max() <= -3
    assert layer.v_mem.min() >= -5

    assert layer.i_syn.max() <= 1.0
    assert layer.i_syn.min() >= 0.0
