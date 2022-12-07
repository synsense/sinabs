import matplotlib.pyplot as plt
import torch

import sinabs
import sinabs.layers as sl


def plot_evolution(neuron_model: sinabs.layers, input: torch.Tensor):
    neuron_model.reset_states()
    output = neuron_model(input)

    plt.figure(figsize=(10, 3))
    for key, recording in neuron_model.recordings.items():
        plt.plot(recording.detach().flatten(), drawstyle="steps", label=key)
    plt.plot(
        output.detach().flatten(), drawstyle="steps", color="black", label="output"
    )
    if not "spike_threshold" in neuron_model.recordings:
        plt.plot(
            [neuron_model.spike_threshold] * input.shape[1], label="spike_threshold"
        )
    plt.xlabel("time")
    plt.title(f"{neuron_model.__class__.__name__} neuron dynamics")
    plt.legend()
