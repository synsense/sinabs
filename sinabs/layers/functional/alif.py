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
    min_v_mem: float,
    b0: float,
):
    # generate spikes and adjust v_mem
    input_tensors = [state[name] for name in activation_fn.spike_fn.required_states]
    spikes = activation_fn.spike_fn.apply(
        *input_tensors, state["threshold"], activation_fn.surrogate_grad_fn
    )

    # Decay the spike threshold and add adaptation factor to it.
    state["b"] = alpha_adapt * state["b"] + (1 - alpha_adapt) * spikes
    state["threshold"] = b0 + adapt_scale * state["b"]

    # if t_syn was provided, we're going to use synaptic current dynamics
    if alpha_syn:
        state["i_syn"] = alpha_syn * (state["i_syn"] + input_data)
    else:
        state["i_syn"] = input_data

    # Decay the membrane potential and add the input currents which are normalised by tau
    state["v_mem"] = alpha_mem * state["v_mem"] + (1 - alpha_mem) * input_data

    state = activation_fn.reset_fn(spikes, state, state["threshold"])

    # Clip membrane potential that is too low
    if min_v_mem is not None:
        state["v_mem"] = (
            torch.nn.functional.relu(state["v_mem"] - min_v_mem) + min_v_mem
        )

    return spikes, state


def alif_forward(
    input_data: torch.Tensor,
    alpha_mem: torch.Tensor,
    alpha_adapt: torch.Tensor,
    alpha_syn: torch.Tensor,
    adapt_scale: float,
    state: dict,
    activation_fn: Callable,
    min_v_mem: float,
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
            min_v_mem=min_v_mem,
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
    min_v_mem: float,
    rec_connect: torch.nn.Module,
    b0: float,
):
    batch_size, n_time_steps, *trailing_dim = input_data.shape

    output_spikes = []
    rec_out = torch.zeros((batch_size, *trailing_dim), device=input_data.device)
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
            min_v_mem=min_v_mem,
            b0=b0,
        )
        output_spikes.append(spikes)

        # compute recurrent output that will be added to the input at the next time step
        rec_out = rec_connect(spikes).reshape((batch_size, *trailing_dim))

    return torch.stack(output_spikes, 1), state
