import torch
import torch.nn as nn
from sinabs.layers import ALIF, ALIFSqueeze, ALIFRecurrent, ALIFRecurrentSqueeze
import pytest
import numpy as np


def test_alif_basic():
    batch_size = 10
    time_steps = 30
    tau_mem = torch.tensor(30)
    alpha = torch.exp(-1/tau_mem)
    input_current = torch.rand(batch_size, time_steps, 2, 7, 7)  / (1-alpha)
    layer = ALIF(tau_mem=tau_mem, threshold=0.1, tau_adapt=tau_mem, adapt_scale=1.8)
    spike_output = layer(input_current)

    assert input_current.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0

def test_alif_squeezed():
    batch_size = 10
    time_steps = 100
    tau_mem = torch.tensor(30)
    alpha = torch.exp(-1/tau_mem)
    input_current = torch.rand(batch_size*time_steps, 2, 7, 7) / (1-alpha)
    layer = ALIFSqueeze(tau_mem=tau_mem, threshold=1, tau_adapt=tau_mem, batch_size=batch_size)
    spike_output = layer(input_current)
    
    assert input_current.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0

def test_alif_minimum_spike_threshold():
    batch_size = 10
    time_steps = 100
    threshold = 10
    tau_mem = torch.tensor(30)
    alpha = torch.exp(-1/tau_mem)
    input_current = torch.zeros(batch_size, time_steps, 2, 7, 7)
    layer = ALIF(tau_mem=tau_mem, threshold=threshold, tau_adapt=tau_mem,)
    spike_output = layer(input_current)

    assert (layer.b_0 + layer.adapt_scale + layer.b >= threshold).all(), "Spike thresholds should not drop below initital threshold."

# def test_alif_spike_threshold_decay():
#     batch_size = 10
#     time_steps = 100
#     threshold = 1
#     alpha = torch.tensor(0.995)
#     adapt_scale = float(1/(1-alpha))
#     input_current = torch.zeros(batch_size, time_steps, 2, 7, 7)
#     input_current[:,0,:,:] = 1 / (1-alpha)# only inject current in the first time step
#     layer = ALIF(tau_mem=tau_mem, tau_adapt=tau_mem, threshold=threshold, adapt_scale=adapt_scale)
#     spike_output = layer(input_current)

#     assert (layer.threshold > threshold).all()
#     # decay only starts after 2 time steps: current integration and adaption
#     threshold_decay = alpha ** (time_steps-2)
#     # account for rounding errors with .isclose()
#     assert torch.isclose(layer.threshold-threshold, threshold_decay, atol=1e-08).all(), "Neuron spike thresholds do not seems to decay correctly."

def test_alif_zero_grad():
    torch.autograd.set_detect_anomaly(False)
    batch_size = 7
    time_steps = 100
    n_neurons = 20
    threshold = 100
    tau_mem = torch.tensor(30)
    alpha = torch.exp(-1/tau_mem)
    sl = ALIF(tau_mem=tau_mem, threshold=threshold, tau_adapt=tau_mem)
    conv = torch.nn.Conv1d(
        in_channels=time_steps,
        out_channels=time_steps,
        kernel_size=3,
        padding=1,
        groups=time_steps,
    )
    sl_0 = ALIF(tau_mem=tau_mem, threshold=threshold, tau_adapt=tau_mem)
    conv_0 = torch.nn.Conv1d(
        in_channels=time_steps,
        out_channels=time_steps,
        kernel_size=3,
        padding=1,
        groups=time_steps,
    )
    model = torch.nn.Sequential(conv, sl)

    # Copy of the original model, where zero_grad will already be applied at beginning
    model_zg = torch.nn.Sequential(conv_0, sl_0)
    model_zg[0].weight.data = model[0].weight.data.clone()
    model_zg[0].bias.data = model[0].bias.data.clone()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    optimizer_zg = torch.optim.SGD(model_zg.parameters(), lr=0.001)

    sl_0.zero_grad()

    data0, data1, data2 = torch.rand((3, batch_size, time_steps, n_neurons))
    out0 = model(data0)
    out0_zg = model_zg(data0)

    loss = torch.nn.functional.mse_loss(out0, torch.ones_like(out0))
    loss_zg = torch.nn.functional.mse_loss(out0_zg, torch.ones_like(out0_zg))
    loss.backward()
    loss_zg.backward()

    grads = [p.grad.data.clone() for p in model.parameters()]
    grads_zg = [p.grad.data.clone() for p in model_zg.parameters()]
    
    for g, g0 in zip(grads, grads_zg):
        assert torch.isclose(g, g0).all()

    optimizer.step()
    optimizer.zero_grad()

    # Detach state gradients to avoid backpropagating through stored states.
    sl.zero_grad()

    out1 = model(data1)

    loss = torch.nn.functional.mse_loss(out1, torch.ones_like(out1))
    loss.backward()

    # Make sure that without detaching there is a RuntimeError
    with pytest.raises(RuntimeError):
        out2 = model(data2)

        loss = torch.nn.functional.mse_loss(out2, torch.ones_like(out2))
        loss.backward()
        
def test_alif_recurrent():
    batch_size = 5
    time_steps = 100
    tau_mem = torch.tensor(30)
    alpha = torch.exp(-1/tau_mem)
    input_dimensions = (batch_size, time_steps, 2, 10)
    n_neurons = np.product(input_dimensions[2:])
    input_current = torch.ones(*input_dimensions) * 0.5 / (1-alpha)
    
    rec_connectivity = nn.Sequential(nn.Flatten(), nn.Linear(n_neurons, n_neurons, bias=False))
    rec_connectivity[1].weight = nn.Parameter(torch.ones(n_neurons,n_neurons)/n_neurons*0.5/(1-alpha))
    layer = ALIFRecurrent(tau_mem=tau_mem, tau_adapt=tau_mem, rec_connectivity=rec_connectivity)
    spike_output = layer(input_current)

    assert input_current.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0

def test_alif_recurrent_squeezed():
    batch_size = 10
    time_steps = 100
    tau_mem = torch.tensor(30)
    alpha = torch.exp(-1/tau_mem)
    input_dimensions = (batch_size*time_steps, 2, 7, 7)
    n_neurons = np.product(input_dimensions[1:])
    input_current = torch.rand(*input_dimensions) / (1-alpha)

    rec_connectivity = nn.Sequential(nn.Flatten(), nn.Linear(n_neurons, n_neurons, bias=False))
    rec_connectivity[1].weight = nn.Parameter(torch.ones(n_neurons,n_neurons)/n_neurons*0.5/(1-alpha))
    layer = ALIFRecurrentSqueeze(tau_mem=tau_mem, tau_adapt=tau_mem, threshold=1, batch_size=batch_size, rec_connectivity=rec_connectivity)
    spike_output = layer(input_current)

    assert input_current.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0