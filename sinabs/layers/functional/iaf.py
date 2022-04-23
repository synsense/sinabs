import torch
from typing import Optional, Callable


def iaf_forward_single(
        input_data: torch.Tensor,
        state: dict,
        spike_threshold: float,
        spike_fn: Callable,
        reset_fn: Callable,
        surrogate_grad_fn: Callable,
        min_v_mem: Optional[float],
):
    # Decay the membrane potential and add the input currents which are normalised by tau
    state["v_mem"] = state["v_mem"] + input_data

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

def iaf_forward(
        input_data: torch.Tensor,
        state: dict,
        spike_threshold: float,
        spike_fn: Callable,
        reset_fn: Callable,
        surrogate_grad_fn: Callable,
        min_v_mem: float,
        record_states: bool = False,
):
    n_time_steps = input_data.shape[1]
    state_names = list(state.keys())

    output_spikes = []
    recordings = []
    for step in range(n_time_steps):
        spikes, state = iaf_forward_single(
            input_data=input_data[:, step],
            state=state,
            spike_threshold=spike_threshold,
            spike_fn=spike_fn,
            reset_fn=reset_fn,
            surrogate_grad_fn=surrogate_grad_fn,
            min_v_mem=min_v_mem,
        )
        output_spikes.append(spikes)
        if record_states:
            recordings.append(state)

    record_dict = {}
    if record_states:
        for state_name in state_names:
            record_dict[state_name] = torch.stack(
                [item[state_name].detach() for item in recordings], 1
            )
    return torch.stack(output_spikes, 1), state, record_dict


def iaf_recurrent(
        input_data: torch.Tensor,
        state: dict,
        spike_threshold: float,
        spike_fn: Callable,
        reset_fn: Callable,
        surrogate_grad_fn: Callable,
        min_v_mem: Optional[float],
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

        spikes, state = iaf_forward_single(
            input_data=total_input,
            state=state,
            spike_threshold=spike_threshold,
            spike_fn=spike_fn,
            reset_fn=reset_fn,
            surrogate_grad_fn=surrogate_grad_fn,
            min_v_mem=min_v_mem,
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
                [item[state_name].detach() for item in recordings], 1
            )
    return torch.stack(output_spikes, 1), state, record_dict
