"""
=============
MultiGaussian
=============
:class:`~sinabs.activation.MultiGaussian` surrogate gradient.
"""

import torch
import sinabs.activation as sina
import matplotlib.pyplot as plt

x = torch.linspace(-0.5, 3.5, 500)
plt.plot(x, sina.MultiGaussian()(v_mem=x, spike_threshold=1.0), label="MultiGaussian")
plt.xlabel("Neuron membrane potential")
plt.ylabel("Derivative")
plt.legend()
