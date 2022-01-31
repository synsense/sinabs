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
