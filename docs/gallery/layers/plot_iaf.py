"""
========================
Integrate and Fire (IAF)
========================
:class:`~sinabs.layers.IAF` layer.
"""

from utils import plot_evolution
import sinabs.layers as sl
import torch


const_current = torch.ones((1, 100, 1)) * 0.03
single_current = torch.zeros((1, 100, 1))
single_current[:, 0] = 0.1

iaf_neuron = sl.IAF(record_states=True)
plot_evolution(iaf_neuron, const_current)


iaf_neuron = sl.IAF(tau_syn=15.0, record_states=True)
plot_evolution(iaf_neuron, single_current)
