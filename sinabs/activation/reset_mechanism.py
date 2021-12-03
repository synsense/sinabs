def membrane_reset(spikes, state, threshold):
    state['v_mem'] = 0
    return state
    
def membrane_subtract(spikes, state, threshold):
    state['v_mem'] = state['v_mem'] - spikes * threshold
    return state
