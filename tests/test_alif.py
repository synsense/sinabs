import torch
import torch.nn as nn
from sinabs.layers import ALIF, ALIFRecurrent
import pytest
import numpy as np


def test_alif_basic():
    batch_size, time_steps = 10, 100
    tau_mem = torch.as_tensor(30.0)
    alpha = torch.exp(-1 / tau_mem)
    input_current = torch.rand(batch_size, time_steps, 2, 7, 7) / (1 - alpha)
    layer = ALIF(tau_mem=tau_mem, tau_adapt=tau_mem, adapt_scale=1.8)
    spike_output = layer(input_current)

    assert layer.does_spike
    assert input_current.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0


def test_alif_basic_audio_input():
    batch_size, time_steps = 10, 100
    tau_mem = torch.as_tensor(30.0)
    alpha = torch.exp(-1 / tau_mem)
    input_current = torch.rand(batch_size, time_steps, 64) / (1 - alpha)
    layer = ALIF(tau_mem=tau_mem, tau_adapt=tau_mem, adapt_scale=1.8)
    spike_output = layer(input_current)

    assert input_current.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0


def test_alif_minimum_spike_threshold():
    batch_size, time_steps = 10, 100
    tau_mem = torch.as_tensor(30.0)
    alpha = torch.exp(-1 / tau_mem)
    input_current = torch.rand(batch_size, time_steps, 2, 7, 7) / (1 - alpha)
    layer = ALIF(
        tau_mem=tau_mem,
        tau_adapt=tau_mem,
    )
    spike_output = layer(input_current)

    assert (
        layer.spike_threshold + layer.b >= 1.0
    ).all(), "Spike thresholds should not drop below initital threshold."
    assert spike_output.sum() > 0
    assert (
        layer.spike_threshold + layer.b > 1.0
    ).all(), "Spike thresholds should be above default threshold if any spikes occured."


def test_alif_spike_threshold_decay():
    batch_size, time_steps = 10, 100
    tau_mem = torch.as_tensor(30.0)
    alpha = torch.exp(-1 / tau_mem)
    spike_threshold = 1.0
    input_current = torch.zeros(batch_size, time_steps, 2, 7, 7)
    input_current[:, 0] = spike_threshold / (
        1 - alpha
    )  # only inject current in the first time step and make it spike
    layer = ALIF(tau_mem=tau_mem, tau_adapt=tau_mem, adapt_scale=1 / (1 - alpha))
    spike_output = layer(input_current)
    layer.reset_states()
    layer.b.fill_(0.0)
    spike_output = layer(input_current)

    assert (layer.spike_threshold > spike_threshold).all()
    assert (
        spike_output.sum()
        == torch.prod(torch.as_tensor(input_current.size())) / time_steps
    ), "All neurons should spike exactly once."
    # decay only starts after first time step
    threshold_decay = alpha ** (time_steps - 1)
    # account for rounding errors with .isclose()
    assert torch.isclose(
        layer.spike_threshold - spike_threshold, threshold_decay, atol=1e-08
    ).all(), "Neuron spike thresholds do not seems to decay correctly."


def test_alif_with_current_dynamics():
    batch_size, time_steps = 10, 100
    tau_mem = torch.as_tensor(30.0)
    tau_syn = torch.tensor(10.0)
    alpha = torch.exp(-1 / tau_mem)
    input_current = torch.rand(batch_size, time_steps, 2, 7, 7) / (1 - alpha)
    layer = ALIF(tau_mem=tau_mem, tau_syn=tau_syn, tau_adapt=tau_mem)
    spike_output = layer(input_current)

    assert input_current.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0


def test_alif_train_alphas():
    batch_size, time_steps = 10, 100
    tau_mem = torch.as_tensor(30.0)
    alpha = torch.exp(-1 / tau_mem)
    input_current = torch.rand(batch_size, time_steps, 2, 7, 7) / (1 - alpha)
    layer = ALIF(tau_mem=tau_mem, train_alphas=True, tau_adapt=tau_mem)
    spike_output = layer(input_current)

    assert input_current.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0


def test_alif_train_alphas_with_current_dynamics():
    batch_size, time_steps = 10, 100
    tau_mem = torch.as_tensor(30.0)
    tau_syn = torch.tensor(10.0)
    alpha = torch.exp(-1 / tau_mem)
    input_current = torch.rand(batch_size, time_steps, 2, 7, 7) / (1 - alpha)
    layer = ALIF(tau_mem=tau_mem, tau_syn=tau_syn, tau_adapt=tau_mem, train_alphas=True)
    spike_output = layer(input_current)

    assert input_current.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0


def test_alif_recurrent():
    batch_size, time_steps = 10, 100
    tau_mem = torch.as_tensor(30.0)
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
    layer = ALIFRecurrent(tau_mem=tau_mem, tau_adapt=tau_mem, rec_connect=rec_connect)
    spike_output = layer(input_current)

    assert input_current.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0


def test_alif_v_mem_recordings():
    batch_size, time_steps = 10, 100
    input_current = torch.rand(batch_size, time_steps, 2, 7, 7)
    layer = ALIF(tau_mem=20.0, tau_adapt=10.0, norm_input=False, record_states=True)
    spike_output = layer(input_current)

    assert layer.recordings["v_mem"].shape == spike_output.shape
    assert not layer.recordings["v_mem"].requires_grad
    assert layer.recordings["spike_threshold"].shape == spike_output.shape
    assert not layer.recordings["spike_threshold"].requires_grad
    assert "i_syn" not in layer.recordings.keys()


def test_alif_i_syn_recordings():
    batch_size, time_steps = 10, 100
    input_current = torch.rand(batch_size, time_steps, 2, 7, 7)
    layer = ALIF(
        tau_mem=20.0, tau_adapt=10.0, tau_syn=10.0, norm_input=False, record_states=True
    )
    spike_output = layer(input_current)

    assert layer.recordings["v_mem"].shape == spike_output.shape
    assert layer.recordings["i_syn"].shape == spike_output.shape
    assert not layer.recordings["v_mem"].requires_grad
    assert not layer.recordings["i_syn"].requires_grad


def test_alif_recurrent_v_mem_recordings():
    batch_size, time_steps, n_neurons = 10, 100, 20
    input_current = torch.rand(batch_size, time_steps, n_neurons)
    rec_connect = nn.Sequential(
        nn.Flatten(), nn.Linear(n_neurons, n_neurons, bias=False)
    )
    layer = ALIFRecurrent(
        tau_mem=20.0,
        tau_adapt=10.0,
        rec_connect=rec_connect,
        norm_input=False,
        record_states=True,
    )
    spike_output = layer(input_current)

    assert layer.recordings["v_mem"].shape == spike_output.shape
    assert not layer.recordings["v_mem"].requires_grad
    assert "i_syn" not in layer.recordings.keys()


def test_alif_recurrent_i_syn_recordings():
    batch_size, time_steps, n_neurons = 10, 100, 20
    input_current = torch.rand(batch_size, time_steps, n_neurons)
    rec_connect = nn.Sequential(
        nn.Flatten(), nn.Linear(n_neurons, n_neurons, bias=False)
    )
    layer = ALIFRecurrent(
        tau_mem=20.0,
        tau_adapt=10.0,
        tau_syn=10.0,
        rec_connect=rec_connect,
        norm_input=False,
        record_states=True,
    )
    spike_output = layer(input_current)

    assert layer.recordings["v_mem"].shape == spike_output.shape
    assert layer.recordings["i_syn"].shape == spike_output.shape
    assert not layer.recordings["v_mem"].requires_grad
    assert not layer.recordings["i_syn"].requires_grad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_alif_on_gpu():
    batch_size, time_steps = 10, 100
    tau_mem = torch.as_tensor(30.0)
    alpha = torch.exp(-1 / tau_mem)
    input_current = torch.rand(batch_size, time_steps, 2, 7, 7) / (1 - alpha)
    layer = ALIF(tau_mem=tau_mem, tau_adapt=tau_mem)

    layer = layer.to("cuda")
    layer.reset_states()
    layer(input_current.to("cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_alif_recurrent_on_gpu():
    batch_size, time_steps, n_neurons = 10, 100, 5
    tau_mem = torch.as_tensor(30.0)
    alpha = torch.exp(-1 / tau_mem)
    input_current = torch.zeros((batch_size, time_steps, n_neurons))
    input_current[:, 0] = 1 / (1 - alpha)

    rec_connect = nn.Linear(n_neurons, n_neurons, bias=False)
    layer = ALIFRecurrent(tau_mem=tau_mem, tau_adapt=tau_mem, rec_connect=rec_connect)

    layer = layer.to("cuda")
    layer(input_current.to("cuda"))
