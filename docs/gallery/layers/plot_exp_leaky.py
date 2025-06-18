"""
=================================
Exponential Leaky Layer (ExpLeak)
=================================
:class:`~sinabs.layers.ExpLeak` layer.
"""

import torch
import matplotlib.pyplot as plt
import sinabs.layers as sl

def plot_evolution(neuron_model, input: torch.Tensor):
    neuron_model.reset_states()
    output = neuron_model(input)
    plt.figure(figsize=(10, 3))
    for key, recording in neuron_model.recordings.items():
        plt.plot(recording.detach().flatten(), drawstyle="steps", label=key)
    plt.plot(
        output.detach().flatten(), drawstyle="steps", color="black", label="output"
    )
    if "spike_threshold" not in neuron_model.recordings:
        plt.plot(
            [neuron_model.spike_threshold] * input.shape[1], label="spike_threshold"
        )
    plt.xlabel("time")
    plt.title(f"{neuron_model.__class__.__name__} neuron dynamics")
    plt.legend()

const_current = torch.ones((1, 100, 1)) * 0.03

exp_leak_neuron = sl.ExpLeak(tau_mem=60.0, record_states=True)
plot_evolution(exp_leak_neuron, const_current)
