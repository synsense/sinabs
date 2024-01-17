from itertools import product

import numpy as np
import pytest
import torch
import torch.nn as nn

import sinabs.activation as sa
from sinabs.layers import LIF, LIFRecurrent


def test_lif_basic():
    torch.set_printoptions(precision=10)
    batch_size, time_steps = 10, 100
    tau_mem = torch.tensor(20.0)
    alpha = torch.exp(-1 / tau_mem)
    input_current = torch.rand(batch_size, time_steps, 2, 7, 7) / (1 - alpha)
    layer = LIF(tau_mem=tau_mem, min_v_mem=-2.0)
    spike_output = layer(input_current)

    # Make sure __repr__ works
    repr(layer)

    # Make sure arg_dict works
    layer.arg_dict

    assert "min_v_mem" in layer.state_dict().keys()
    assert "spike_threshold" in layer.state_dict().keys()
    assert layer.does_spike
    assert input_current.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0


def test_lif_array_tau():
    torch.set_printoptions(precision=10)
    batch_size, time_steps = 10, 100
    # Shape for tau mem such that 2nd dimension needs to be expanded
    tau_mem = torch.randn(2, 1, 7) + 20.0
    tau_syn = torch.randn(2, 1, 7) + 20.0
    alpha = torch.exp(-1 / tau_mem)
    input_current = torch.rand(batch_size, time_steps, 2, 7, 7) / (1 - alpha)
    layer = LIF(tau_mem=tau_mem, tau_syn=tau_syn)
    spike_output = layer(input_current)

    assert input_current.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0


def test_lif_v_mem_recordings():
    batch_size, time_steps = 10, 100
    input_current = torch.rand(batch_size, time_steps, 2, 7, 7)
    input_current[:, :5] = 0
    layer = LIF(tau_mem=20.0, norm_input=False, record_states=True)
    spike_output = layer(input_current)

    layer.recordings["v_mem"].sum().backward()
    assert layer.tau_mem.grad is not None

    assert layer.recordings["v_mem"].shape == spike_output.shape
    # Ensure causality
    assert (layer.recordings["v_mem"][:, :5] == 0).all()
    assert "i_syn" not in layer.recordings.keys()


def test_lif_i_syn_recordings():
    batch_size, time_steps = 10, 100
    input_current = torch.rand(batch_size, time_steps, 2, 7, 7)
    layer = LIF(tau_mem=20.0, tau_syn=10.0, norm_input=False, record_states=True)
    spike_output = layer(input_current)

    layer.recordings["i_syn"].sum().backward()
    assert layer.tau_syn.grad is not None

    assert layer.recordings["v_mem"].shape == spike_output.shape
    assert layer.recordings["i_syn"].shape == spike_output.shape


def test_lif_recurrent_v_mem_recordings():
    batch_size, time_steps, n_neurons = 10, 100, 20
    input_current = torch.rand(batch_size, time_steps, n_neurons)
    rec_connect = nn.Sequential(
        nn.Flatten(), nn.Linear(n_neurons, n_neurons, bias=False)
    )
    layer = LIFRecurrent(
        tau_mem=20.0, rec_connect=rec_connect, norm_input=False, record_states=True
    )
    spike_output = layer(input_current)

    layer.recordings["v_mem"].sum().backward()
    assert layer.tau_mem.grad is not None

    assert layer.recordings["v_mem"].shape == spike_output.shape
    assert "i_syn" not in layer.recordings.keys()


def test_lif_recurrent_i_syn_recordings():
    batch_size, time_steps, n_neurons = 10, 100, 20
    input_current = torch.rand(batch_size, time_steps, n_neurons)
    rec_connect = nn.Sequential(
        nn.Flatten(), nn.Linear(n_neurons, n_neurons, bias=False)
    )
    layer = LIFRecurrent(
        tau_mem=20.0,
        tau_syn=10.0,
        rec_connect=rec_connect,
        norm_input=False,
        record_states=True,
    )
    spike_output = layer(input_current)

    layer.recordings["i_syn"].sum().backward()
    assert layer.tau_syn.grad is not None

    assert layer.recordings["v_mem"].shape == spike_output.shape
    assert layer.recordings["i_syn"].shape == spike_output.shape


def test_lif_single_spike():
    torch.set_printoptions(precision=10)
    batch_size, time_steps = 10, 100
    tau_mem = torch.tensor(20.0)
    input_current = torch.rand(batch_size, time_steps, 2, 7, 7) * 20
    layer = LIF(tau_mem=tau_mem, spike_fn=sa.SingleSpike)
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
    input_current = torch.rand(batch_size, time_steps, 2, 7, 7) * 100
    layer = LIF(tau_mem=tau_mem, spike_fn=sa.MaxSpike(max_spikes))
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
    tau_mem = 20.0
    input_current = torch.zeros((batch_size, time_steps, n_neurons))
    input_current[:, 0] = 1

    layer = LIF(tau_mem=tau_mem, norm_input=False)
    spikes = layer(input_current)

    assert layer.firing_rate > 0
    assert layer.firing_rate == spikes.sum() / (batch_size * time_steps * n_neurons)

@pytest.mark.parametrize("tau_syn", [20.0, None])
def test_lif_norm_input_with_synapse(tau_syn):
    batch_size, time_steps, n_neurons = 5, 10, 5
    # Different tau_mem between 10 and 100
    tau_mem = (torch.rand(n_neurons) * 90) + 10
    input_current = torch.rand((batch_size, time_steps, n_neurons))
    # Very high spike threshold to prevent model from spiking (we need pre-spike v_mem)
    spike_threshold = 1e6

    layer = LIF(
        spike_threshold=spike_threshold,
        tau_syn=tau_syn,
        tau_mem=tau_mem,
        norm_input=False,
        record_states=True,
    )
    layer_norm = LIF(
        spike_threshold=spike_threshold,
        tau_syn=tau_syn,
        tau_mem=tau_mem,
        norm_input=True,
        record_states=True,
    )
    spikes = layer(input_current)
    spikes_norm = layer_norm(input_current)

    alpha_mem = torch.exp(-1.0 / tau_mem)
    normalization_factor = 1 - alpha_mem

    assert torch.allclose(
        layer.recordings["v_mem"] * normalization_factor, layer_norm.recordings["v_mem"]
    )
    if tau_syn is not None:
        # synaptic current should not be affected by normalization
        assert (layer.recordings["i_syn"] == layer_norm.recordings["i_syn"]).all()


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


def test_min_v_mem():
    batch_size, time_steps = 10, 1
    tau_mem = torch.tensor(30.0)
    alpha = torch.exp(-1 / tau_mem)
    input_data = torch.rand(batch_size, time_steps, 2, 7, 7) / -(1 - alpha)
    layer = LIF(tau_mem=tau_mem)
    layer(input_data)
    assert (layer.v_mem < -0.5).any()

    layer = LIF(tau_mem=tau_mem, min_v_mem=-0.5)
    layer(input_data)
    assert not (layer.v_mem < -0.5).any()


params = product((20.0, None), (True, False))


@pytest.mark.parametrize("tau_syn,train_alphas", params)
def test_alpha_tau_conversion(tau_syn, train_alphas):
    tau_mem = torch.rand((3, 4)) * 20 + 30
    alpha_mem = torch.exp(-1.0 / tau_mem)

    layer = LIF(tau_mem=tau_mem, tau_syn=tau_syn, train_alphas=train_alphas)
    assert torch.isclose(layer.tau_mem_calculated, tau_mem).all()
    assert torch.isclose(layer.alpha_mem_calculated, alpha_mem).all()
    if tau_syn is None:
        assert layer.tau_syn_calculated is None
        assert layer.alpha_syn_calculated is None
    else:
        tau_syn = torch.as_tensor(tau_syn)
        alpha_syn = torch.exp(-1.0 / tau_syn)
        assert torch.isclose(layer.tau_syn_calculated, tau_syn).all()
        assert torch.isclose(layer.alpha_syn_calculated, alpha_syn).all()
