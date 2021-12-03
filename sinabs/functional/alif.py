import torch
from typing import Callable


def alif_forward(
    input_data: torch.Tensor,
    alpha_mem: torch.Tensor,
    alpha_adapt: torch.Tensor,
    alpha_syn: torch.Tensor,
    adapt_scale: float,
    state: dict,
    activation_fn: Callable,
    threshold_low: float,
):
    batch_size, time_steps, *trailing_dim = input_data.shape

    output_spikes = []
    for step in range(time_steps):
        # if t_syn was provided, we're going to use synaptic current dynamics
        if alpha_syn:
            state['i_syn'] = alpha_syn * state['i_syn'] + input_data[:, step]
        else:
            state['i_syn'] = input_data[:, step]

        # Decay the membrane potential and add the input currents which are normalised by tau
        state['v_mem'] = alpha_mem * state['v_mem'] + (1 - alpha_mem) * input_data[:, step]

        # Clip membrane potential that is too low
        if threshold_low:
            state['v_mem'] = torch.nn.functional.relu(state['v_mem'] - threshold_low) + threshold_low

        # generate spikes and adjust v_mem
        spikes, state = activation_fn(state)
        output_spikes.append(spikes)

        self.v_mem = state['v_mem']
        self.i_syn = state['i_syn'] if alpha_syn else None
        # Decay the spike threshold and add adaptation factor to it.
        self.b = alpha_adapt * self.b + (1 - alpha_adapt) * spikes
        self.threshold = self.b_0 + self.adapt_scale*self.b

        output_spikes.append(self.activations)