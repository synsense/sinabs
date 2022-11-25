"""
=============
MultiGaussian
=============
:class:`~sinabs.activation.MultiGaussian` surrogate gradient.
"""

import matplotlib.pyplot as plt
import torch

import sinabs.activation as sina

x = torch.linspace(-2, 4, 500)
plt.plot(x, sina.MultiGaussian()(v_mem=x, spike_threshold=1), label="MultiGaussian")
plt.xlabel("Neuron membrane potential")
plt.ylabel("Derivative")
plt.legend()
plt.show()
