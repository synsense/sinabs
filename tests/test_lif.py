import torch
import torch.nn as nn
from sinabs.layers import LIF, LIFRecurrent
import sinabs.activation as sa
import numpy as np
import pytest


def test_lif_basic():
    torch.set_printoptions(precision=10)
    batch_size, time_steps = 10, 100
    tau_mem = torch.tensor(20.0)
    alpha = torch.exp(-1 / tau_mem)
    input_current = torch.rand(batch_size, time_steps, 2, 7, 7) / (1 - alpha)
    layer = LIF(tau_mem=tau_mem)
    spike_output = layer(input_current)

    assert input_current.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0


def test_lif_single_spike():
    torch.set_printoptions(precision=10)
    batch_size, time_steps = 10, 100
    tau_mem = torch.tensor(20.0)
    activation_fn = sa.ActivationFunction(spike_fn=sa.SingleSpike)
    input_current = torch.rand(batch_size, time_steps, 2, 7, 7) * 20
    layer = LIF(tau_mem=tau_mem, activation_fn=activation_fn)
    spike_output = layer(input_current)

    assert input_current.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0
    assert torch.max(spike_output) == 1


def test_lif_max_spike():
    torch.set_printoptions(precision=10)
    batch_size, time_steps = 10, 100
    tau_mem = torch.tensor(20.0)
    max_spikes = 3
    activation_fn = sa.ActivationFunction(spike_fn=sa.MaxSpike(max_spikes))
    input_current = torch.rand(batch_size, time_steps, 2, 7, 7) * 100
    layer = LIF(tau_mem=tau_mem, activation_fn=activation_fn)
    spike_output = layer(input_current)

    assert input_current.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0
    assert torch.max(spike_output) == max_spikes


def test_lif_with_current_dynamics():
    batch_size, time_steps = 10, 100
    tau_mem = torch.tensor(30.0)
    tau_syn = torch.tensor(10.0)
    alpha = torch.exp(-1 / tau_mem)
    input_current = torch.rand(batch_size, time_steps, 2, 7, 7) / (1 - alpha)
    layer = LIF(tau_mem=tau_mem, tau_syn=tau_syn)
    spike_output = layer(input_current)

    assert input_current.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0


def test_lif_train_alphas():
    batch_size, time_steps = 10, 100
    tau_mem = torch.tensor(30.0)
    alpha = torch.exp(-1 / tau_mem)
    input_current = torch.rand(batch_size, time_steps, 2, 7, 7) / (1 - alpha)
    layer = LIF(tau_mem=tau_mem, train_alphas=True)
    spike_output = layer(input_current)

    assert input_current.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0


def test_lif_train_alphas_with_current_dynamics():
    batch_size, time_steps = 10, 100
    tau_mem = torch.tensor(30.0)
    tau_syn = torch.tensor(10.0)
    alpha = torch.exp(-1 / tau_mem)
    input_current = torch.rand(batch_size, time_steps, 2, 7, 7) / (1 - alpha)
    layer = LIF(tau_mem=tau_mem, tau_syn=tau_syn, train_alphas=True)
    spike_output = layer(input_current)

    assert input_current.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0


def test_lif_input_integration():
    batch_size, time_steps = 10, 100
    tau_mem = torch.tensor(30.0)
    alpha = torch.exp(-1 / tau_mem)
    input_current = torch.zeros(batch_size, time_steps, 2, 7, 7)
    # only inject current in the first time step
    input_current[:, 0] = 1 / (1 - alpha)
    layer = LIF(tau_mem=tau_mem)
    spike_output = layer(input_current)

    assert (
        spike_output.sum() == np.product(input_current.shape) / time_steps
    ), "Every neuron should spike exactly once."
    assert (
        spike_output[:, 0].sum() == spike_output.sum()
    ), "First output time step should contain all the spikes."


def test_lif_membrane_decay():
    batch_size, time_steps = 10, 100
    tau_mem = torch.tensor(30.0)
    start_value = 0.75
    alpha = torch.exp(-1 / tau_mem)
    input_current = torch.zeros(batch_size, time_steps, 2, 7, 7)
    # only inject current in the first time step
    input_current[:, 0] = start_value / (1 - alpha)
    layer = LIF(tau_mem=tau_mem)
    spike_output = layer(input_current)

    # decay only starts after first time step
    membrane_decay = start_value * alpha ** (time_steps - 1)
    # account for rounding errors with .isclose()
    assert torch.isclose(
        layer.v_mem, membrane_decay, atol=1e-08
    ).all(), "Neuron membrane potentials do not seems to decay correctly."


def test_lif_recurrent_basic():
    batch_size, time_steps = 10, 100
    tau_mem = torch.tensor(30.0)
    alpha = torch.exp(-1 / tau_mem)
    input_dimensions = (batch_size, time_steps, 2, 10)
    n_neurons = np.product(input_dimensions[2:])
    input_current = torch.ones(*input_dimensions) * 0.5 / (1 - alpha)

    rec_connect = nn.Sequential(
        nn.Flatten(), nn.Linear(n_neurons, n_neurons, bias=False)
    )
    rec_connect[1].weight = nn.Parameter(
        torch.ones(n_neurons, n_neurons) / n_neurons * 0.5 / (1 - alpha)
    )
    layer = LIFRecurrent(tau_mem=tau_mem, rec_connect=rec_connect)
    spike_output = layer(input_current)

    assert input_current.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() == batch_size * (time_steps - 2) * 2 * 10


def test_lif_with_shape():
    batch_size, time_steps, n_neurons = 10, 100, 5
    tau_mem = torch.tensor(30.0)
    alpha = torch.exp(-1 / tau_mem)
    input_current = torch.zeros((batch_size, time_steps, n_neurons))
    input_current[:, 0] = 1 / (1 - alpha)

    layer = LIF(tau_mem=tau_mem, shape=(batch_size, n_neurons))
    v_mem_shape_intitial = layer.v_mem.shape
    spike_output = layer(input_current)

    assert layer.v_mem.shape == v_mem_shape_intitial
    assert input_current.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() == batch_size * n_neurons


def test_lif_with_multiple_taus():
    batch_size, time_steps, n_neurons = 2, 2, 5
    tau_mem = torch.tensor([i * 10.0 for i in range(1, 6)])
    alpha = torch.exp(-1 / tau_mem)
    input_current = torch.zeros((batch_size, time_steps, n_neurons))
    # make sure that v_mem after the first ts is beneath spike threshold
    v_mem = 0.75
    input_current[:, 0] = v_mem / (1 - alpha)

    layer = LIF(tau_mem=tau_mem)
    layer(input_current)

    assert torch.allclose(layer.v_mem[0], alpha * v_mem)
    assert (layer.v_mem[0] == layer.v_mem[1]).all()

def test_lif_firing_rate():
    batch_size, time_steps, n_neurons = 5, 10, 5
    tau_mem = 20.
    input_current = torch.zeros((batch_size, time_steps, n_neurons))
    input_current[:, 0] = 1

    layer = LIF(tau_mem=tau_mem, norm_input=False)
    spikes = layer(input_current)

    assert layer.firing_rate > 0
    assert layer.firing_rate == spikes.sum() / (batch_size * time_steps * n_neurons)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_lif_on_gpu():
    batch_size, time_steps = 5, 20
    tau_mem = torch.as_tensor(30.0)
    alpha = torch.exp(-1 / tau_mem)
    input_current = torch.rand(batch_size, time_steps, 2, 7, 7) / (1 - alpha)
    layer = LIF(tau_mem=tau_mem)

    layer = layer.to("cuda")
    layer(input_current.to("cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_lif_recurrent_on_gpu():
    batch_size, time_steps = 5, 20
    tau_mem = torch.as_tensor(30.0)
    alpha = torch.exp(-1 / tau_mem)
    input_dimensions = (batch_size, time_steps, 2, 10)
    n_neurons = np.product(input_dimensions[2:])
    input_current = torch.ones(*input_dimensions) * 0.5 / (1 - alpha)

    rec_connect = nn.Sequential(
        nn.Flatten(), nn.Linear(n_neurons, n_neurons, bias=False)
    )
    layer = LIFRecurrent(tau_mem=tau_mem, rec_connect=rec_connect)

    layer = layer.to("cuda")
    layer(input_current.to("cuda"))


def test_threshold_low():
    batch_size, time_steps = 10, 1
    tau_mem = torch.tensor(30.0)
    alpha = torch.exp(-1 / tau_mem)
    input_data = torch.rand(batch_size, time_steps, 2, 7, 7)/ -(1 - alpha)
    layer = LIF(tau_mem=tau_mem)
    layer(input_data)
    assert (layer.v_mem < -0.5).any()

    layer = LIF(tau_mem=tau_mem, threshold_low=-0.5)
    layer(input_data)
    assert not (layer.v_mem < -0.5).any()