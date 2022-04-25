import torch
import torch.nn as nn
from sinabs.layers import IAF, IAFSqueeze, IAFRecurrent
import sinabs.activation as sa
import pytest
import numpy as np


def test_iaf_basic():
    batch_size, time_steps = 10, 100
    input_current = torch.rand(batch_size, time_steps, 2, 7, 7)
    layer = IAF()
    spike_output = layer(input_current)

    assert layer.does_spike
    assert input_current.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0


def test_iaf_v_mem_recordings():
    batch_size, time_steps = 10, 100
    input_current = torch.rand(batch_size, time_steps, 2, 7, 7)
    layer = IAF(record_states=True)
    spike_output = layer(input_current)

    assert layer.recordings["v_mem"].shape == spike_output.shape
    assert not layer.recordings["v_mem"].requires_grad
    assert "i_syn" not in layer.recordings.keys()

# # This test doesn't make sense anymore
# def test_iaf_i_syn_recordings():
#     batch_size, time_steps = 10, 100
#     input_current = torch.rand(batch_size, time_steps, 2, 7, 7)
#     layer = IAF(tau_syn=10.0, record_states=True)
#     spike_output = layer(input_current)
#
#     assert layer.recordings["v_mem"].shape == spike_output.shape
#     assert layer.recordings["i_syn"].shape == spike_output.shape
#     assert not layer.recordings["v_mem"].requires_grad
#     assert not layer.recordings["i_syn"].requires_grad


def test_iaf_recurrent_v_mem_recordings():
    batch_size, time_steps, n_neurons = 10, 100, 20
    input_current = torch.rand(batch_size, time_steps, n_neurons)
    rec_connect = nn.Sequential(
        nn.Flatten(), nn.Linear(n_neurons, n_neurons, bias=False)
    )
    layer = IAFRecurrent(rec_connect=rec_connect, record_states=True)
    spike_output = layer(input_current)

    assert layer.recordings["v_mem"].shape == spike_output.shape
    assert not layer.recordings["v_mem"].requires_grad
    assert "i_syn" not in layer.recordings.keys()


def test_iaf_recurrent_i_syn_recordings():
    batch_size, time_steps, n_neurons = 10, 100, 20
    input_current = torch.rand(batch_size, time_steps, n_neurons)
    rec_connect = nn.Sequential(
        nn.Flatten(), nn.Linear(n_neurons, n_neurons, bias=False)
    )
    layer = IAFRecurrent(rec_connect=rec_connect, record_states=True)
    spike_output = layer(input_current)

    assert layer.recordings["v_mem"].shape == spike_output.shape
    assert not layer.recordings["v_mem"].requires_grad


def test_iaf_single_spike():
    batch_size, time_steps = 10, 100
    layer = IAF(spike_fn=sa.SingleSpike)
    input_current = torch.rand(batch_size, time_steps, 2, 7, 7) * 100
    spike_output = layer(input_current)

    assert input_current.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0
    assert torch.max(spike_output) == 1


def test_iaf_max_spike():
    batch_size, time_steps = 10, 100
    max_spikes = 3
    layer = IAF(spike_fn=sa.MaxSpike(max_spikes))
    input_current = torch.rand(batch_size, time_steps, 2, 7, 7) * 100
    spike_output = layer(input_current)

    assert input_current.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0
    assert torch.max(spike_output) == max_spikes


def test_iaf_squeezed():
    batch_size, time_steps = 10, 100
    input_current = torch.rand(batch_size * time_steps, 2, 7, 7)
    layer = IAFSqueeze(batch_size=batch_size)
    spike_output = layer(input_current)

    assert input_current.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0


def test_iaf_recurrent():
    batch_size, time_steps = 10, 100
    input_dimensions = (batch_size, time_steps, 2, 10)
    n_neurons = np.product(input_dimensions[2:])
    input_current = torch.ones(*input_dimensions) * 0.5

    rec_connect = nn.Sequential(
        nn.Flatten(), nn.Linear(n_neurons, n_neurons, bias=False)
    )
    rec_connect[1].weight = nn.Parameter(
        torch.ones(n_neurons, n_neurons) / n_neurons * 0.5
    )
    layer = IAFRecurrent(rec_connect=rec_connect)
    spike_output = layer(input_current)

    assert input_current.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0


def test_iaf_firing_rate():
    batch_size, time_steps, n_neurons = 5, 10, 5
    input_current = torch.zeros((batch_size, time_steps, n_neurons))
    input_current[:, 0] = 1

    layer = IAF()
    spikes = layer(input_current)

    assert layer.firing_rate == spikes.sum() / (batch_size * time_steps * n_neurons)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_iaf_on_gpu():
    batch_size, time_steps = 10, 100
    input_current = torch.rand(batch_size, time_steps, 2, 7, 7)
    layer = IAF()

    layer = layer.to("cuda")
    layer(input_current.to("cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_iaf_recurrent_on_gpu():
    batch_size, time_steps = 10, 100
    input_dimensions = (batch_size, time_steps, 2, 10)
    n_neurons = np.product(input_dimensions[2:])
    input_current = torch.ones(*input_dimensions)

    rec_connect = nn.Sequential(
        nn.Flatten(), nn.Linear(n_neurons, n_neurons, bias=False)
    )
    layer = IAFRecurrent(rec_connect=rec_connect)

    layer = layer.to("cuda")
    layer(input_current.to("cuda"))


def test_parameters_in_layer():
    layer = IAF()
    # IAF must not have any trainable parameters by default
    assert (len(list(layer.named_parameters())) == 0)


def test_speed_iaf_vs_lif():
    from sinabs.layers import IAF, LIF
    layer_iaf = nn.Sequential(nn.Linear(100, 100, bias=False), IAF())
    layer_lif = nn.Sequential(nn.Linear(100, 100, bias=False), LIF(tau_mem=np.inf, norm_input=False))
    # Copy weights to be identical
    layer_lif[0].weight.data.copy_(layer_iaf[0].weight.data)
    layer_lif[1].tau_mem.requires_grad = False

    input_data = torch.rand((1, 1000, 100))

    import time

    # LIF
    t_start = time.time()
    for i in range(10):
        layer_lif.zero_grad()
        layer_lif[1].reset_states()
        out_lif = layer_lif(input_data)
        out_lif.sum().backward()
    t_stop = time.time()
    time_lif = t_stop - t_start

    # IAF
    t_start = time.time()
    for i in range(10):
        layer_iaf.zero_grad()
        layer_iaf[1].reset_states()
        out_iaf = layer_iaf(input_data)
        out_iaf.sum().backward()
    t_stop = time.time()
    time_iaf = t_stop - t_start

    print(time_iaf, time_lif)
