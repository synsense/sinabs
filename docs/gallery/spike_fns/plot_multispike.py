"""
===========
MultiSpike
===========
:class:`~sinabs.activation.MultiSpike` activation function.
"""

import torch
import sinabs.activation as sina
import matplotlib.pyplot as plt

v_mem = torch.linspace(0, 5.5, 500)

spike_threshold = 1.0
activations = sina.MultiSpike.apply(v_mem, spike_threshold, sina.MultiGaussian())
plt.plot(v_mem, activations)
plt.xlabel("Neuron membrane potential")
plt.ylabel("Spike activation")
plt.tight_layout()
