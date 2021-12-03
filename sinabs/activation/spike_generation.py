import torch


def multi_spike(state, threshold):
    return (state['v_mem'] > 0) * torch.div(state['v_mem'], threshold, rounding_mode="trunc").float()

def single_spike(state, threshold):
    return (state['v_mem'] - threshold > 0).float()
    