import torch
from typing import Callable


def alif_forward_single(
    input_data: torch.Tensor,
    alpha_mem: torch.Tensor,
    alpha_adapt: torch.Tensor,
    alpha_syn: torch.Tensor,
    adapt_scale: float,
    state: dict,
    activation_fn: Callable,
    threshold_low: float,
    b0: float,
):
    batch_size, time_steps, *trailing_dim = input_data.shape
    
    # if t_syn was provided, we're going to use synaptic current dynamics
    if alpha_syn:
        state['i_syn'] = alpha_syn * (state['i_syn'] + input_data)
    else:
        state['i_syn'] = input_data

    # Decay the membrane potential and add the input currents which are normalised by tau
    state['v_mem'] = alpha_mem * state['v_mem'] + (1 - alpha_mem) * input_data

    # Clip membrane potential that is too low
    if threshold_low:
        state['v_mem'] = torch.nn.functional.relu(state['v_mem'] - threshold_low) + threshold_low

    # generate spikes and adjust v_mem
    spikes, state = activation_fn(state)

    # Decay the spike threshold and add adaptation factor to it.
    state['b'] = alpha_adapt * state['b'] + (1 - alpha_adapt) * spikes
#     breakpoint()
    activation_fn.spike_threshold = b0 + adapt_scale*state['b']

    return spikes, state


def alif_forward(
    input_data: torch.Tensor,
    alpha_mem: torch.Tensor,
    alpha_adapt: torch.Tensor,
    alpha_syn: torch.Tensor,
    adapt_scale: float,
    state: dict,
    activation_fn: Callable,
    threshold_low: float,
    b0: float,
):
    batch_size, time_steps, *trailing_dim = input_data.shape
    
    output_spikes = []
    for step in range(time_steps):
        spikes, state = alif_forward_single(
            input_data=input_data[:, step],
            alpha_mem=alpha_mem,
            alpha_adapt=alpha_adapt,
            alpha_syn=alpha_syn,
            adapt_scale=adapt_scale,
            state=state,
            activation_fn=activation_fn,
            threshold_low=threshold_low,
            b0=b0,
        )
        output_spikes.append(spikes)

    return torch.stack(output_spikes, 1), state


def alif_recurrent(
    input_data: torch.Tensor,
    alpha_mem: torch.Tensor,
    alpha_adapt: torch.Tensor,
    alpha_syn: torch.Tensor,
    adapt_scale: float,
    state: dict,
    activation_fn: Callable,
    threshold_low: float,
    rec_connect: torch.nn.Module,
    b0: float,
):
    batch_size, n_time_steps, *trailing_dim = input_data.shape

    output_spikes = []
    rec_out = torch.zeros((batch_size, *trailing_dim))
    for step in range(n_time_steps):
        total_input = input_data[:, step] + rec_out

        spikes, state = alif_forward_single(
            input_data=total_input,
            alpha_mem=alpha_mem,
            alpha_adapt=alpha_adapt,
            alpha_syn=alpha_syn,
            adapt_scale=adapt_scale,
            state=state,
            activation_fn=activation_fn,
            threshold_low=threshold_low,
            b0=b0,
        )
        output_spikes.append(spikes)
        
        # compute recurrent output that will be added to the input at the next time step
        rec_out = rec_connect(spikes).reshape((batch_size, *trailing_dim))

    return torch.stack(output_spikes, 1), state
