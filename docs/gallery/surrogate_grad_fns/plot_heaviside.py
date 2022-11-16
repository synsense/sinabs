"""
=========
Heaviside
=========
:class:`~sinabs.activation.Heaviside` surrogate gradient.
"""

import matplotlib.pyplot as plt
import torch

import sinabs.activation as sina

x = torch.linspace(-2, 4, 500)
plt.plot(x, sina.Heaviside(window=0.5)(v_mem=x, spike_threshold=1.0), label="Heaviside")
plt.xlabel("Neuron membrane potential")
plt.ylabel("Derivative")
plt.legend()
