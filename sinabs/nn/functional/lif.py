from typing import Callable, Optional

import torch


def lif_forward_single(
    input_data: torch.Tensor,
    alpha_mem: float,
    alpha_syn: float,
    state: dict,
    spike_threshold: float,
    spike_fn: Callable,
    reset_fn: Callable,
    surrogate_grad_fn: Callable,
    min_v_mem: Optional[float],
    norm_input: bool,
):
    # if t_syn was provided, we're going to use synaptic current dynamics
    if alpha_syn is not None:
        state["i_syn"] = alpha_syn * (state["i_syn"] + input_data)
    else:
        state["i_syn"] = input_data

    if norm_input:
        synaptic_input = (1 - alpha_mem) * state["i_syn"]
    else:
        synaptic_input = state["i_syn"]

    # Decay the membrane potential and add the input currents which are normalised by tau
    state["v_mem"] = alpha_mem * state["v_mem"] + synaptic_input

    # generate spikes and adjust v_mem
    if spike_fn:
        input_tensors = [state[name] for name in spike_fn.required_states]
        spikes = spike_fn.apply(*input_tensors, spike_threshold, surrogate_grad_fn)
        state = reset_fn(spikes, state, spike_threshold)
    else:
        spikes = state["v_mem"].clone()
        state = state.copy()

    if min_v_mem is not None:
        state["v_mem"] = (
            torch.nn.functional.relu(state["v_mem"] - min_v_mem) + min_v_mem
        )
    return spikes, state


def lif_forward(
    input_data: torch.Tensor,
    alpha_mem: float,
    alpha_syn: float,
    state: dict,
    spike_threshold: float,
    spike_fn: Callable,
    reset_fn: Callable,
    surrogate_grad_fn: Callable,
    min_v_mem: float,
    norm_input: bool,
    record_states: bool = False,
):
    n_time_steps = input_data.shape[1]
    state_names = list(state.keys())

    output_spikes = []
    if record_states:
        recordings = {name: [] for name in state_names}

    for step in range(n_time_steps):
        spikes, state = lif_forward_single(
            input_data=input_data[:, step],
            alpha_mem=alpha_mem,
            alpha_syn=alpha_syn,
            state=state,
            spike_threshold=spike_threshold,
            spike_fn=spike_fn,
            reset_fn=reset_fn,
            surrogate_grad_fn=surrogate_grad_fn,
            min_v_mem=min_v_mem,
            norm_input=norm_input,
        )
        output_spikes.append(spikes)
        if record_states:
            for name in state_names:
                recordings[name].append(state[name].clone())

    if record_states:
        record_dict = {name: torch.stack(vals, 1) for name, vals in recordings.items()}
    else:
        record_dict = dict()

    return torch.stack(output_spikes, 1), state, record_dict


def lif_recurrent(
    input_data: torch.Tensor,
    alpha_mem: float,
    alpha_syn: float,
    state: dict,
    spike_threshold: float,
    spike_fn: Callable,
    reset_fn: Callable,
    surrogate_grad_fn: Callable,
    min_v_mem: Optional[float],
    norm_input: bool,
    rec_connect: torch.nn.Module,
    record_states: bool = False,
):
    batch_size, n_time_steps, *trailing_dim = input_data.shape
    state_names = list(state.keys())

    output_spikes = []
    recordings = []
    rec_out = torch.zeros((batch_size, *trailing_dim), device=input_data.device)
    for step in range(n_time_steps):
        total_input = input_data[:, step] + rec_out

        spikes, state = lif_forward_single(
            input_data=total_input,
            alpha_mem=alpha_mem,
            alpha_syn=alpha_syn,
            state=state,
            spike_threshold=spike_threshold,
            spike_fn=spike_fn,
            reset_fn=reset_fn,
            surrogate_grad_fn=surrogate_grad_fn,
            min_v_mem=min_v_mem,
            norm_input=norm_input,
        )
        output_spikes.append(spikes)
        if record_states:
            recordings.append(state)

        # compute recurrent output that will be added to the input at the next time step
        rec_out = rec_connect(spikes).reshape((batch_size, *trailing_dim))

    record_dict = {}
    if record_states:
        for state_name in state_names:
            record_dict[state_name] = torch.stack(
                [item[state_name] for item in recordings], 1
            )
    return torch.stack(output_spikes, 1), state, record_dict
