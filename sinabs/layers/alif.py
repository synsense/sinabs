from typing import Optional, Union
import torch
from .spiking_layer import SpikingLayer
from .pack_dims import squeeze_class
from .lif import LIF


__all__ = ["ALIF", "ALIFSqueeze"]

# Learning window for surrogate gradient
window = 1.0


class ALIF(LIF):
    def __init__(
        self,
        tau_mem: Union[float, torch.Tensor],
        tau_threshold: Union[float, torch.Tensor],
        threshold: Union[float, torch.Tensor] = 1.0,
        threshold_low: Union[float, None] = -1.0,
        threshold_adaptation: Union[float, torch.Tensor] = 0.1,
        membrane_subtract: Optional[float] = None,
        membrane_reset=False,
        *args,
        **kwargs,
    ):
        """
        Pytorch implementation of a Leaky Integrate and Fire neuron with threshold apdaption and learning enabled.
        In addition to the LIF neuron mechanics, the firing threshold :math:`\\theta` also adapts in the following way:

        .. math ::
            \\frac{d\\theta}{dt} = - \\frac{\\theta - \\theta _{0}}{\\tau_{\\theta}}

            \\text{if } V_m(t) = V_{th} \\text{, then } \\theta \\rightarrow \\theta + \\alpha

        where :math:`alpha` corresponds to the `threshold_adaptation` and :math:`\\theta` to the `threshold` parameter.

        Parameters
        ----------
        tau_mem: float
            Membrane potential decay time constant.
        tau_threshold: float
            Spike threshold decay time constant.
        threshold: float
            Spiking threshold of the neuron.
        threshold_low: float or None
            Lower bound for membrane potential.
        threshold_adaption: float
            The amount that the spike threshold is bumped up for every spike, after which it decays back to the initial threshold.
        membrane_subtract: float or None
            The amount to subtract from the membrane potential upon spiking.
            Default is equal to threshold. Ignored if membrane_reset is set.
        membrane_reset: bool
            If True, reset the membrane to 0 on spiking.
        """
        super().__init__(
            tau_mem=tau_mem,
            threshold=threshold,
            threshold_low=threshold_low,
            membrane_subtract=membrane_subtract,
            membrane_reset=membrane_reset,
            *args,
            **kwargs,
        )
        self.tau_threshold = torch.tensor(tau_threshold)
        self.threshold_adaptation = torch.tensor(threshold_adaptation)
        self.resting_threshold = self.threshold
        delattr(self, 'threshold')
        self.register_buffer("threshold", torch.zeros(1))

    def check_states(self, input_current):
        """ Initialise spike threshold states when the first input is received. """
        super().check_states(input_current)
        if self.threshold.shape == torch.Size([1]):
            self.threshold = torch.ones_like(input_current[:,0]) * self.resting_threshold

    def update_state_after_spike(self):
        """ Use LIF's update_state_after_spike method as a hook to update neuron threshold states. """
        super().update_state_after_spike()
        self.adapt_threshold_state(self.activations)

    def adapt_threshold_state(self, output_spikes):
        """ Decay the spike threshold and add adaption constant to it. """
        self.threshold = self.threshold - self.resting_threshold
        beta = torch.exp(-1.0/self.tau_threshold)
        self.threshold = self.threshold * beta
        self.threshold = self.threshold + output_spikes * self.threshold_adaptation + self.resting_threshold

    def reset_states(self, shape=None, randomize=False):
        """ Reset the state of all neurons and threshold states in this layer. """
        super().reset_states(shape, randomize)
        if shape is None:
            shape = self.threshold.shape
        if randomize:
            self.threshold = torch.rand(shape, device=self.threshold.device)
        else:
            self.threshold = torch.ones(shape, device=self.threshold.device) * self.resting_threshold

ALIFSqueeze = squeeze_class(ALIF)
