"""
========================================
Adaptive Leaky Integrate and Fire (ALIF)
========================================
:class:`~sinabs.layers.ALIF` layer with an adaptive threshold based on the output spikes.
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

const_current = torch.ones((1, 100, 1)) * 0.12

alif_neuron = sl.ALIF(
    tau_mem=40.0, tau_adapt=40.0, adapt_scale=20, norm_input=False, record_states=True
)
plot_evolution(alif_neuron, const_current)
