import torch
from typing import Optional

def lif_forward_single(
    input_data: torch.Tensor,
    alpha_mem: float,
    alpha_syn: float,
    state: dict,
    activation_fn,
    threshold_low: Optional[float],
    norm_input: bool,
):
    # if t_syn was provided, we're going to use synaptic current dynamics
    if alpha_syn:
        state["i_syn"] = alpha_syn * (state["i_syn"] + input_data)
    else:
        state["i_syn"] = input_data

    if norm_input:
        state["i_syn"] = (1 - alpha_mem) * state["i_syn"]

    # Decay the membrane potential and add the input currents which are normalised by tau
    state["v_mem"] = alpha_mem * state["v_mem"] + state["i_syn"]

    # generate spikes and adjust v_mem
    spikes, state = activation_fn(state)

    if threshold_low is not None:
        state["v_mem"] = (
                torch.nn.functional.relu(state["v_mem"] - threshold_low) + threshold_low
        )
    return spikes, state


def lif_forward(
    input_data: torch.Tensor,
    alpha_mem: float,
    alpha_syn: float,
    state: dict,
    activation_fn,
    threshold_low: float,
    norm_input: bool,
):
    n_time_steps = input_data.shape[1]

    output_spikes = []
    for step in range(n_time_steps):
        spikes, state = lif_forward_single(
            input_data=input_data[:, step],
            alpha_mem=alpha_mem,
            alpha_syn=alpha_syn,
            state=state,
            activation_fn=activation_fn,
            threshold_low=threshold_low,
            norm_input=norm_input,
        )
        output_spikes.append(spikes)

    return torch.stack(output_spikes, 1), state


def lif_recurrent(
    input_data: torch.Tensor,
    alpha_mem: float,
    alpha_syn: float,
    state: dict,
    activation_fn,
    threshold_low: Optional[float],
    norm_input: bool,
    rec_connect: torch.nn.Module,
):
    batch_size, n_time_steps, *trailing_dim = input_data.shape

    output_spikes = []
    rec_out = torch.zeros((batch_size, *trailing_dim), device=input_data.device)
    for step in range(n_time_steps):
        total_input = input_data[:, step] + rec_out

        spikes, state = lif_forward_single(
            input_data=total_input,
            alpha_mem=alpha_mem,
            alpha_syn=alpha_syn,
            state=state,
            activation_fn=activation_fn,
            threshold_low=threshold_low,
            norm_input=norm_input,
        )
        output_spikes.append(spikes)

        # compute recurrent output that will be added to the input at the next time step
        rec_out = rec_connect(spikes).reshape((batch_size, *trailing_dim))

    return torch.stack(output_spikes, 1), state
