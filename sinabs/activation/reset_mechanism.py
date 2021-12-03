from dataclasses import dataclass, field


@dataclass
class MembraneReset:
    def __call__(self, spikes, state, threshold):
        state['v_mem'] = 0
        return state

@dataclass
class MembraneSubtract:
    def __call__(self, spikes, state, threshold):
        state['v_mem'] = state['v_mem'] - spikes * threshold
        return state
