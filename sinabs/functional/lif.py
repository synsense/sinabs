import torch


def lif_forward(
    input_data: torch.Tensor,
    alpha_mem: float,
    alpha_syn: float,
    state: dict,
    activation_fn,
    threshold_low,
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
        state['v_mem'] = alpha_mem * state['v_mem'] + (1 - alpha_mem) * state['i_syn']

        # Clip membrane potential that is too low
        if threshold_low:
            state['v_mem'] = torch.nn.functional.relu(state['v_mem'] - threshold_low) + threshold_low

        # generate spikes and adjust v_mem
        spikes, state = activation_fn(state)
        output_spikes.append(spikes)

    return torch.stack(output_spikes, 1), state
