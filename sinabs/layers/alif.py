from typing import Optional, Union
import torch
from .spiking_layer import SpikingLayer
from .pack_dims import squeeze_class
from .lif import LIF
from .functional import ThresholdSubtract, ThresholdReset


__all__ = ["ALIF", "ALIFSqueeze"]

# Learning window for surrogate gradient
window = 1.0


class ALIF(SpikingLayer):
    def __init__(
        self,
        alpha_mem: Union[float, torch.Tensor],
        alpha_adapt: Union[float, torch.Tensor],
        adapt_scale: Union[float, torch.Tensor] = 1.8,
        threshold: Union[float, torch.Tensor] = 1.0,
        membrane_reset: bool = False,
        threshold_low: Optional[float] = None,
        membrane_subtract: Optional[float] = None,
        *args,
        **kwargs,
    ):
        """
        Pytorch implementation of a Long Short Term Memory SNN (LSNN) by Bellec et al., 2018:
        https://papers.neurips.cc/paper/2018/hash/c203d8a151612acf12457e4d67635a95-Abstract.html

        In addition to the LIF neuron mechanics, the firing threshold :math:`\\theta` also adapts in the following way:

        .. math ::
            \\frac{d\\theta}{dt} = - \\frac{\\theta - \\theta _{0}}{\\tau_{\\theta}}

            \\text{if } V_m(t) = V_{th} \\text{, then } \\theta \\rightarrow \\theta + \\alpha

        where :math:`alpha` corresponds to the `adaptation` and :math:`\\theta` to the `threshold` parameter.

        Parameters
        ----------
        alpha_mem: float
            Membrane potential decay time constant.
        alpha_adapt: float
            Spike threshold decay time constant.
        adaption: float
            The amount that the spike threshold is bumped up for every spike, after which it decays back to the initial threshold.
        threshold: float
            Spiking threshold of the neuron.
        threshold_low: float or None
            Lower bound for membrane potential.
        membrane_reset: bool
            If True, reset the membrane to 0 on spiking.
        membrane_subtract: float or None
            The amount to subtract from the membrane potential upon spiking.
            Default is equal to threshold. Ignored if membrane_reset is set.
        """
        super().__init__(
            threshold=threshold,
            threshold_low=threshold_low,
            membrane_subtract=membrane_subtract,
            membrane_reset=membrane_reset,
            *args,
            **kwargs,
        )
        self.alpha_mem = alpha_mem
        self.alpha_adapt = alpha_adapt
        self.adapt_scale = adapt_scale
        self.b_0 = threshold
        self.register_buffer("b", torch.zeros(1))
        self.reset_function = ThresholdReset if membrane_reset else ThresholdSubtract

    def check_states(self, input_current):
        """Initialise spike threshold states when the first input is received."""
        shape_without_time = (input_current.shape[0], *input_current.shape[2:])
        if self.state.shape != shape_without_time:
            self.reset_states(shape=shape_without_time, randomize=False)

    def forward(self, input_current: torch.Tensor):
        """
        Forward pass with given data.

        Parameters
        ----------
        input_current : torch.Tensor
            Data to be processed. Expected shape: (batch, time, ...)

        Returns
        -------
        torch.Tensor
            Output data. Same shape as `input_spikes`.
        """
        # Ensure the neuron state are initialized
        self.check_states(input_current)

        time_steps = input_current.shape[1]
        output_spikes = torch.zeros_like(input_current)

        for step in range(time_steps):
            # Decay the membrane potential
            self.state = self.state * self.alpha_mem

            # Add the input currents which are normalised by tau to membrane potential state
            self.state = self.state + (1 - self.alpha_mem) * input_current[:, step]

            # Clip membrane potential that is too low
            if self.threshold_low:
                self.state = torch.clamp(self.state, min=self.threshold_low)

            # generate spikes
            activations = self.reset_function.apply(
                self.state,
                self.b * self.adapt_scale + self.b_0,
                (self.b * self.adapt_scale + self.b_0) * window,
            )
            output_spikes[:, step] = activations

            self.state = self.state - activations * (
                self.b_0 + self.adapt_scale * self.b
            )

            # Decay the spike threshold and add adaption constant to it.
            self.b = self.alpha_adapt * self.b + (1 - self.alpha_adapt) * activations
            print(self.b.mean())

        self.tw = time_steps
        self.spikes_number = output_spikes.abs().sum()
        return output_spikes

    def reset_states(self, shape=None, randomize=False):
        """Reset the state of all neurons and threshold states in this layer."""
        super().reset_states(shape, randomize)
        if shape is None:
            shape = self.b.shape
        if randomize:
            self.b = torch.rand(shape, device=self.b.device)
        else:
            self.b = torch.zeros(shape, device=self.b.device)

    @property
    def _param_dict(self) -> dict:
        param_dict = super()._param_dict
        param_dict.update(
            alpha_mem=self.alpha_mem,
            alpha_adapt=self.alpha_adapt,
            adapt_scale=self.adapt_scale,
        )

        return param_dict


ALIFSqueeze = squeeze_class(ALIF)
