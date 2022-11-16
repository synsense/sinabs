"""
=================================
Exponential Leaky Layer (ExpLeak)
=================================
:class:`~sinabs.layers.ExpLeak` layer.
"""

import torch
from utils import plot_evolution

import sinabs.layers as sl

const_current = torch.ones((1, 100, 1)) * 0.03


exp_leak_neuron = sl.ExpLeak(tau_mem=60.0, record_states=True)
plot_evolution(exp_leak_neuron, const_current)
