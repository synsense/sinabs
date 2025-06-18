"""
==============================
Leaky Integrate and Fire (LIF)
==============================
The :class:`~sinabs.layers.LIF` layer. This neuron integrates the input and decays its state at every time step.
"""

import torch
import matplotlib.pytlot as plt
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

lif_neuron = sl.LIF(tau_mem=40.0, norm_input=False, record_states=True)
plot_evolution(lif_neuron, const_current)


# By default, no synaptic dynamics are used. We can enable that by setting tau_syn. Note that instead of a constant current, we now provide input only at the first time step.

lif_neuron = sl.LIF(tau_mem=40.0, tau_syn=30.0, norm_input=False, record_states=True)
plot_evolution(lif_neuron, single_current)
