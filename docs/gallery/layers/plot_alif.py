"""
========================================
Adaptive Leaky Integrate and Fire (ALIF)
========================================
:class:`~sinabs.layers.ALIF` layer with an adaptive threshold based on the output spikes. 
"""

import torch
from utils import plot_evolution

import sinabs.layers as sl

const_current = torch.ones((1, 100, 1)) * 0.12

alif_neuron = sl.ALIF(
    tau_mem=40.0, tau_adapt=40.0, adapt_scale=20, norm_input=False, record_states=True
)
plot_evolution(alif_neuron, const_current)
