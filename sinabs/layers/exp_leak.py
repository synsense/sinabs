from typing import Optional, Union

import torch

from .lif import LIF
from .reshape import SqueezeMixin


class ExpLeak(LIF):
    """Leaky Integrator layer which is a special case of :class:`~sinabs.layers.LIF` without
    activation function.

    Neuron dynamics in discrete time:

    .. math ::
        V_{mem}(t+1) = \\alpha V_{mem}(t) + (1-\\alpha)\\sum z(t)

    where :math:`\\alpha =  e^{-1/tau_{mem}}` and :math:`\\sum z(t)` represents the sum of all input currents at time :math:`t`.

    Args:
        tau_mem: Membrane potential time constant.
        min_v_mem: Lower bound for membrane potential v_mem, clipped at every time step.
        train_alphas: When True, the discrete decay factor exp(-1/tau) is used for training rather than tau itself.
        shape: Optionally initialise the layer state with given shape. If None, will be inferred from input_size.
        norm_input: When True, normalise input current by tau. This helps when training time constants.
        record_states: When True, will record all internal states such as v_mem or i_syn in a dictionary attribute `recordings`. Default is False.
    """

    def __init__(
        self,
        tau_mem: Union[float, torch.Tensor],
        shape: Optional[torch.Size] = None,
        train_alphas: bool = False,
        min_v_mem: Optional[float] = None,
        norm_input: bool = False,
        record_states: bool = False,
    ):
        super().__init__(
            tau_mem=tau_mem,
            tau_syn=None,
            spike_threshold=None,
            train_alphas=train_alphas,
            min_v_mem=min_v_mem,
            shape=shape,
            spike_fn=None,
            reset_fn=None,
            surrogate_grad_fn=None,
            norm_input=norm_input,
            record_states=record_states,
        )

    @property
    def _param_dict(self) -> dict:
        param_dict = super()._param_dict
        param_dict.pop("tau_syn")
        param_dict.pop("spike_fn")
        param_dict.pop("reset_fn")
        param_dict.pop("surrogate_grad_fn")
        param_dict.pop("spike_threshold")
        return param_dict


class ExpLeakSqueeze(ExpLeak, SqueezeMixin):
    """ExpLeak layer with 4-dimensional input (Batch*Time, Channel, Height, Width).

    Same as parent ExpLeak class, only takes in squeezed 4D input (Batch*Time, Channel, Height, Width)
    instead of 5D input (Batch, Time, Channel, Height, Width) in order to be compatible with
    layers that can only take a 4D input, such as convolutional and pooling layers.

    Shape:
        - Input: :math:`(Batch \\times Time, Channel, Height, Width)` or :math:`(Batch \\times Time, Channel)`
        - Output: Same as input.

    Attributes:
        v_mem: The membrane potential resets according to reset_fn for every spike.
        i_syn: This attribute is only available if tau_syn is not None.
    """

    def __init__(self, batch_size=None, num_timesteps=None, **kwargs):
        super().__init__(**kwargs)
        self.squeeze_init(batch_size, num_timesteps)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Forward call wrapper that will flatten the input to and unflatten the output from the
        super class forward call."""
        return self.squeeze_forward(input_data, super().forward)

    @property
    def _param_dict(self) -> dict:
        return self.squeeze_param_dict(super()._param_dict)
