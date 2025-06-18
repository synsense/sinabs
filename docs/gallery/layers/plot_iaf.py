"""
========================
Integrate and Fire (IAF)
========================
:class:`~sinabs.layers.IAF` layer.
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
single_current = torch.zeros((1, 100, 1))
single_current[:, 0] = 0.1

iaf_neuron = sl.IAF(record_states=True)
plot_evolution(iaf_neuron, const_current)


iaf_neuron = sl.IAF(tau_syn=15.0, record_states=True)
plot_evolution(iaf_neuron, single_current)
