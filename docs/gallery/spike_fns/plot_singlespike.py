"""
===========
SingleSpike
===========
:class:`~sinabs.activation.SingleSpike` activation function.
"""

import matplotlib.pyplot as plt
import torch

import sinabs.activation as sina

v_mem = torch.linspace(0, 5.5, 500)

spike_threshold = 1.0
activations = sina.SingleSpike.apply(v_mem, spike_threshold, sina.MultiGaussian())
plt.plot(v_mem, activations)
plt.xlabel("Neuron membrane potential")
plt.ylabel("Spike activation")
plt.ylim(top=5.2)
plt.tight_layout()
