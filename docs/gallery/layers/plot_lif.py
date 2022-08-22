"""
==============================
Leaky Integrate and Fire (LIF)
==============================
The :class:`~sinabs.layers.LIF` layer. This neuron integrates the input and decays its state at every time step.
"""

from utils import plot_evolution
import sinabs.layers as sl
import torch


const_current = torch.ones((1, 100, 1)) * 0.03
single_current = torch.zeros((1, 100, 1))
single_current[:, 0] = 0.1

lif_neuron = sl.LIF(tau_mem=40.0, norm_input=False, record_states=True)
plot_evolution(lif_neuron, const_current)


# By default, no synaptic dynamics are used. We can enable that by setting tau_syn. Note that instead of a constant current, we now provide input only at the first time step.

lif_neuron = sl.LIF(tau_mem=40.0, tau_syn=30.0, norm_input=False, record_states=True)
plot_evolution(lif_neuron, single_current)
